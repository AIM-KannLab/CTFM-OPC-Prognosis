"""A1 driver: clinical-only vs CT-FM-ST vs combined-A, few-shot and full-shot.

Three arms, all scored on the SAME held-out test set, which is re-derived per
endpoint with the published seed-42 split:

  clinical_only  age, sex, HPV, smoking, T_Stage, N_Stage        (Cox)
  ctfm_st        512 frozen CT-FM features                        (ridge Cox, lambda on val)
  combined_A     6 clinical + 1 out-of-fold imaging risk score    (Cox)

With A1_IMAGING=segres_ft the imaging arm is segres_ft instead -- SegResEncoder trained
from scratch, read from the networks train_segres.py trained -- and combined_A uses its
out-of-fold score. Everything else is identical.

Shot sizes come from `subsample.shot_grid`, which reproduces the grid the published
image-only script actually ran (Radcure 10..940, Maastro 10..121).

Results are written PER CELL as soon as a cell finishes, so a run can be interrupted
and resumed, and cells can be run in parallel across tmux panes.

  python run_a1.py                                   # every cohort x endpoint
  python run_a1.py --cohort Radcure --endpoint os    # one cell
  python run_a1.py --force                           # recompute cells already done
  python run_a1.py --spec ajcc                       # AJCC sensitivity spec

Run `extract_features.py` once first.
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2

from bootstrap_ci import bootstrap_ci, cindex, paired_bootstrap
from cohort import (COHORTS, DATA_VERSION, FEATURE_VERSION, IMAGING, IMAGING_LABEL, NSTAGE,
                    new_run_id, results_dir, tee_output, write_run_info, ENDPOINTS,
                    ENDPOINT_PAPER_NAME, SPEC_AJCC, SPEC_PRIMARY, build_cohort,
                    split_dataset_ratio)
from extract_features import load_features
from ipcw_brier import eval_times_for, ibs_from_probs
from models import (dl_score_independent_test, dl_score_wald,
                    fit_clinical_and_combined, fit_ctfm_st, fit_segres_ft)
from subsample import SEEDS, shot_grid, stratified_subsample

RESULTS = None        # set in main(): results_<version>_<run id>/
CELLS = None
ARMS = ["clinical_only", IMAGING, "combined_A"]          # IMAGING: ctfm_st or segres_ft
B_BOOT = 2000

# seed sentinels used in the `seed` column
FULL_SHOT = -1        # single deterministic fit on the whole training split
POOLED = -2           # few-shot, seeds pooled, bootstrap CI over test patients


def _metrics(res, times, events, eval_times, boot: bool):
    """C-index and IBS, with bootstrap 95% CIs when `boot`."""
    out = {"cindex": np.nan, "cindex_lo": np.nan, "cindex_hi": np.nan,
           "ibs": np.nan, "ibs_lo": np.nan, "ibs_hi": np.nan}
    if res.failed or not np.isfinite(res.risk_test).all():
        return out
    risk = res.risk_test
    if boot:
        p, lo, hi, _ = bootstrap_ci(
            lambda i: cindex(times[i], events[i], risk[i]), times, events, B=B_BOOT)
        out.update(cindex=p, cindex_lo=lo, cindex_hi=hi)
    else:
        out["cindex"] = cindex(times, events, risk)
    if res.surv_test is not None and np.isfinite(res.surv_test).all():
        sv = res.surv_test
        fn = lambda i: ibs_from_probs(sv[i], times[i], events[i], eval_times)
        if boot:
            p, lo, hi, _ = bootstrap_ci(fn, times, events, B=B_BOOT)
            out.update(ibs=p, ibs_lo=lo, ibs_hi=hi)
        else:
            out["ibs"] = fn(np.arange(len(times)))
    return out


DL_COLS = ["dl_hr", "dl_lo", "dl_hi", "dl_p", "dl_penalized",
           "dl_test_hr", "dl_test_lo", "dl_test_hi", "dl_test_p"]


def _dl_tests(fits, times, events):
    """DL-score significance for one fitted trio of models.

    w  Wald test of the DL score inside the combined model (training data)
    t  frozen clinical + DL scores re-tested together on the held-out patients
    """
    w = dl_score_wald(fits.get("combined_A"))
    cl, im = fits.get("clinical_only"), fits.get(IMAGING)
    t = None
    if cl is not None and im is not None and not cl.failed and not im.failed:
        t = dl_score_independent_test(times, events, cl.risk_test, im.risk_test)
    return w, t


def _dl_cols(arm, w, t):
    """DL columns for a results row -- filled only on the combined_A row, the one
    model that has the DL score as an input."""
    d = dict.fromkeys(DL_COLS, np.nan)
    if arm == "combined_A":
        if w is not None:
            d.update(dl_hr=w[0], dl_lo=w[1], dl_hi=w[2], dl_p=w[3], dl_penalized=w[4])
        if t is not None:
            d.update(dl_test_hr=t[0], dl_test_lo=t[1], dl_test_hi=t[2], dl_test_p=t[3])
    return d


def _fit_all(tr, va, te, f_tr, f_va, f_te, cols, eval_times, seed, cohort, endpoint):
    """Fit the three arms on one training set. combined_A reuses the imaging stage-1 fit."""
    if IMAGING == "segres_ft":
        imaging, stage1 = fit_segres_ft(tr, va, te, eval_times, cohort, endpoint, seed)
    else:
        imaging, stage1 = fit_ctfm_st(tr, va, te, f_tr, f_va, f_te, eval_times), None
    clin, comb = fit_clinical_and_combined(tr, va, te, f_tr, f_va, f_te, cols,
                                           eval_times, imaging, seed, stage1=stage1)
    return {"clinical_only": clin, IMAGING: imaging, "combined_A": comb}


def _save_preds(arm, cohort, endpoint, spec, tag, **arrays):
    d = RESULTS / arm / "predictions"
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(d / f"{cohort}_{endpoint}_{spec}_{tag}.npz", **arrays)


def _atomic_csv(df: pd.DataFrame, path: Path) -> None:
    """Write via a temp file + rename, so another job can never read a half-written
    cell (a rename is atomic: the file is either absent or complete)."""
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


REGIMES = ("full", "few")


def _cell_files(cohort, endpoint, spec, regime):
    """Full-shot and few-shot results are separate files, so each can finish alone."""
    return (CELLS / f"rows_{regime}_{cohort}_{endpoint}_{spec}.csv",
            CELLS / f"comps_{regime}_{cohort}_{endpoint}_{spec}.csv")


def _regime_done(cohort, endpoint, spec, regime) -> bool:
    r, c = _cell_files(cohort, endpoint, spec, regime)
    if r.exists() and c.exists():
        return True
    # runs made before the split stored both regimes in one file per cell
    return ((CELLS / f"rows_{cohort}_{endpoint}_{spec}.csv").exists()
            and (CELLS / f"comps_{cohort}_{endpoint}_{spec}.csv").exists())


def _write_cell(rows, comps, cohort, endpoint, spec, regime) -> None:
    CELLS.mkdir(parents=True, exist_ok=True)
    r, c = _cell_files(cohort, endpoint, spec, regime)
    # rows first, comps last: a regime counts as finished only once BOTH exist
    _atomic_csv(pd.DataFrame(rows), r)
    _atomic_csv(pd.DataFrame(comps), c)
    print(f"   [{regime}-shot saved -> {r.name}]")


def _maybe_build_report(spec: str, run_id: str) -> None:
    """Build or update the report as results complete.

    Called by every job when it finishes. Once all full-shot cells exist the report is
    built (full-shot tables and figures); once the few-shot cells exist too it is
    rebuilt with the few-shot tables and learning curves. A lock serialises jobs that
    finish together, and a marker stops the same level from being built twice.
    A1_AUTOREPORT=0 disables it (run_pipeline.sh builds the reports itself).
    """
    if os.environ.get("A1_AUTOREPORT", "1") == "0":
        return
    cells = [(co, ep) for co in COHORTS for ep in ENDPOINTS]
    full = sum(_regime_done(co, ep, spec, "full") for co, ep in cells)
    few = sum(_regime_done(co, ep, spec, "few") for co, ep in cells)
    if full < len(cells):
        print(f"\nfull-shot {full}/{len(cells)}, few-shot {few}/{len(cells)} cells done -- the "
              f"report is built automatically once all full-shot cells exist.\n"
              f"   (to build it now from what exists:  python3 make_report.py --run {run_id})")
        return
    level = "complete" if few == len(cells) else "full-shot"
    import fcntl
    with open(RESULTS / ".report.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)              # wait if another job is building
        marker = RESULTS / (".report_level" if spec == "primary" else f".report_level_{spec}")
        if marker.exists() and marker.read_text().strip() == level:
            print(f"\n{level} report already built by another job -- skipping here")
            return
        print(f"\nbuilding the {level} report"
              + ("" if level == "complete" else f"  (few-shot: {few}/{len(cells)} cells so far)"))
        try:
            import make_report
            make_report.main(folder=str(RESULTS), spec=spec)
            marker.write_text(level)
        except Exception as exc:                  # results are safe on disk regardless
            print(f"!! report failed ({exc}); build it with:  python3 make_report.py --run {run_id} --spec {spec}")


def run_cell(cohort: str, endpoint: str, spec_name: str, cols: list[str],
             regimes=("full", "few")) -> None:
    t0 = time.time()
    rows, comps = [], []

    df = build_cohort(cohort, endpoint)
    tr_full, va, te = split_dataset_ratio(df)
    if IMAGING == "segres_ft":            # no features: the networks' outputs are read instead
        import segres_cache
        segres_cache.check_trained([cohort], [endpoint], regimes)
        get = lambda d: None
    else:
        feat_map = load_features(cohort)
        get = lambda d: np.stack([feat_map[k] for k in d["key"]])
    f_full, f_va, f_te = get(tr_full), get(va), get(te)
    eval_times = eval_times_for(tr_full["time"].values, tr_full["event"].values)
    times, events = te["time"].values, te["event"].values
    grid = shot_grid(len(tr_full))

    base = dict(cohort=cohort, endpoint=ENDPOINT_PAPER_NAME[endpoint],
                endpoint_code=endpoint, spec=spec_name,
                n_test=len(te), n_events=int(events.sum()))
    print(f"\n[{cohort}/{ENDPOINT_PAPER_NAME[endpoint]}] train={len(tr_full)} "
          f"val={len(va)} test={len(te)} ({int(events.sum())} events)")
    print(f"  shot grid: {grid}")

    do_full, do_few = "full" in regimes, "few" in regimes

    # ------------------------------------------------ full shot
    if do_full:
        fits = _fit_all(tr_full, va, te, f_full, f_va, f_te, cols, eval_times, 42,
                        cohort, endpoint)
        w_full, t_full = _dl_tests(fits, times, events)
        for arm, res in fits.items():
            m = _metrics(res, times, events, eval_times, boot=True)
            rows.append({**base, "arm": arm, "regime": "full_shot", "n_train": len(tr_full),
                         "seed": FULL_SHOT, "epv": res.epv, "n_params": res.n_params,
                         "hyperparam": res.hyperparam, "penalized": res.penalized,
                         "failed": res.failed, **m, **_dl_cols(arm, w_full, t_full)})
            print(f"   {arm:<14} C={m['cindex']:.4f} [{m['cindex_lo']:.3f},{m['cindex_hi']:.3f}]"
                  f"  IBS={m['ibs']:.4f} [{m['ibs_lo']:.3f},{m['ibs_hi']:.3f}]"
                  f"  EPV={res.epv:.1f}  {res.hyperparam}"
                  + (f"  FAILED: {res.failed}" if res.failed else ""))
            _save_preds(arm, cohort, endpoint, spec_name, "fullshot",
                        keys=te["key"].values, times=times, events=events,
                        risk=res.risk_test, eval_times=eval_times,
                        surv=res.surv_test if res.surv_test is not None else np.empty((0, 0)))
            if res.model is not None:
                # a reusable model: the fitted Cox model plus everything a new patient's data
                # must go through first (scaler, age standardization, img_score scaling)
                d = RESULTS / arm / "models"; d.mkdir(parents=True, exist_ok=True)
                with open(d / f"{cohort}_{endpoint}_{spec_name}_fullshot.pkl", "wb") as fh:
                    pickle.dump({"model": res.model, "preprocess": res.preprocess,
                                 "arm": arm, "cohort": cohort, "endpoint": endpoint,
                                 "spec": spec_name, "n_train": len(tr_full),
                                 "hyperparam": res.hyperparam, "features": FEATURE_VERSION,
                                 "data": DATA_VERSION}, fh)

        ok = {a: f for a, f in fits.items() if not f.failed and np.isfinite(f.risk_test).all()}
        for a, b in [("combined_A", "clinical_only"), (IMAGING, "clinical_only"),
                     ("combined_A", IMAGING)]:
            if a in ok and b in ok:
                ra, rb = ok[a].risk_test, ok[b].risk_test
                d, lo, hi, p = paired_bootstrap(
                    lambda i: cindex(times[i], events[i], ra[i]),
                    lambda i: cindex(times[i], events[i], rb[i]), times, events, B=B_BOOT)
                comps.append({**base, "regime": "full_shot", "n_train": len(tr_full),
                              "metric": "cindex", "model_a": a, "model_b": b,
                              "delta": d, "lo": lo, "hi": hi, "p": p})
                print(f"   Δ {a} − {b}: {d:+.4f} [{lo:+.3f},{hi:+.3f}] p={p:.4f}")

        ca, cl = fits.get("combined_A"), fits.get("clinical_only")
        if (ca and cl and ca.loglik is not None and cl.loglik is not None
                and not ca.penalized and not cl.penalized):
            stat = 2 * (ca.loglik - cl.loglik)
            pv = float(chi2.sf(max(stat, 0), 1))
            comps.append({**base, "regime": "full_shot", "n_train": len(tr_full),
                          "metric": "LRT", "model_a": "combined_A", "model_b": "clinical_only",
                          "delta": stat, "lo": np.nan, "hi": np.nan, "p": pv})
            print(f"   LRT combined_A vs clinical: chi2={stat:.3f} p={pv:.4g}   "
                  f"(approximate: imaging_score is a derived covariate)")


        # ---- DL-score significance: (1) Wald inside the combined model, training data;
        #      (2) frozen clinical + DL scores re-tested on patients neither was fitted on.
        w = w_full
        if w is not None:
            comps.append({**base, "regime": "full_shot", "n_train": len(tr_full),
                          "metric": "dl_wald_train", "model_a": "combined_A",
                          "model_b": "", "delta": w[0], "lo": w[1], "hi": w[2], "p": w[3],
                          "penalized": w[4]})
            print(f"   DL score in combined model (train): HR/SD {w[0]:.3f} [{w[1]:.2f},{w[2]:.2f}]"
                  f" p={w[3]:.3g}" + ("   (penalized fit: p not valid)" if w[4] else ""))
        t = t_full
        if t is not None:
            comps.append({**base, "regime": "full_shot", "n_train": len(tr_full),
                          "metric": "dl_independent_test", "model_a": "dl_score",
                          "model_b": "clinical_score",
                          "delta": t[0], "lo": t[1], "hi": t[2], "p": t[3]})
            print(f"   DL score re-tested on held-out test set: HR/SD {t[0]:.3f} [{t[1]:.2f},{t[2]:.2f}] p={t[3]:.3g}")

        _write_cell(rows, comps, cohort, endpoint, spec_name, "full")

    # ------------------------------------------------ few shot
    if do_few:
        rows, comps = [], []
        for shot in [s for s in grid if s < len(tr_full)]:
            pool = {a: {"risk": [], "t": [], "e": [], "surv": [], "st": [], "se": []}
                    for a in ARMS}
            summ = {a: [] for a in ARMS}
            dl_seen = []                                     # per-seed DL results, for the log
            for seed in SEEDS:
                sub = stratified_subsample(tr_full, shot, seed)
                fs = _fit_all(sub, va, te, get(sub), f_va, f_te, cols, eval_times, seed,
                              cohort, endpoint)
                w_s, t_s = _dl_tests(fs, times, events)
                dl_seen.append((w_s, t_s))
                for arm, res in fs.items():
                    m = _metrics(res, times, events, eval_times, boot=False)
                    summ[arm].append(m["cindex"])
                    rows.append({**base, "arm": arm, "regime": "few_shot", "n_train": len(sub),
                                 "seed": seed, "epv": res.epv, "n_params": res.n_params,
                                 "hyperparam": res.hyperparam, "penalized": res.penalized,
                                 "failed": res.failed, **m, **_dl_cols(arm, w_s, t_s)})
                    if not res.failed and np.isfinite(res.risk_test).all():
                        P = pool[arm]
                        P["risk"].append(res.risk_test); P["t"].append(times); P["e"].append(events)
                        if res.surv_test is not None and np.isfinite(res.surv_test).all():
                            P["surv"].append(res.surv_test)
                            P["st"].append(times); P["se"].append(events)

            # seeds pooled -> bootstrap 95% CI for BOTH C-index and IBS at this shot size
            pooled_ci = {}
            for arm, P in pool.items():
                if not P["risk"]:
                    continue
                r = np.concatenate(P["risk"]); t = np.concatenate(P["t"]); e = np.concatenate(P["e"])
                cp, clo, chi_ = bootstrap_ci(
                    lambda i: cindex(t[i], e[i], r[i]), t, e, B=B_BOOT)[:3]
                ip = ilo = ihi = np.nan
                if P["surv"]:
                    sv = np.vstack(P["surv"])
                    st = np.concatenate(P["st"]); se = np.concatenate(P["se"])
                    ip, ilo, ihi = bootstrap_ci(
                        lambda i: ibs_from_probs(sv[i], st[i], se[i], eval_times),
                        st, se, B=B_BOOT)[:3]
                pooled_ci[arm] = (clo, chi_)
                rows.append({**base, "arm": arm, "regime": "few_shot_pooled", "n_train": shot,
                             "seed": POOLED, "epv": np.nan, "n_params": np.nan,
                             "hyperparam": "", "penalized": np.nan, "failed": "",
                             "cindex": cp, "cindex_lo": clo, "cindex_hi": chi_,
                             "ibs": ip, "ibs_lo": ilo, "ibs_hi": ihi})
                _save_preds(arm, cohort, endpoint, spec_name, f"fewshot_n{shot}",
                            times=t, events=e, risk=r, eval_times=eval_times,
                            surv=np.vstack(P["surv"]) if P["surv"] else np.empty((0, 0)))

            print("   n=%-5d %s" % (shot, "  ".join(
                f"{a}={np.nanmean(summ[a]):.3f}±{np.nanstd(summ[a]):.3f}"
                + (f"[{pooled_ci[a][0]:.3f},{pooled_ci[a][1]:.3f}]" if a in pooled_ci else "")
                for a in ARMS)))
            ws = [w for w, _ in dl_seen if w is not None]
            ts = [t for _, t in dl_seen if t is not None]
            valid_w = [w for w in ws if not w[4]]            # unpenalized fits only
            train_p = (f"p<0.05 in {sum(w[3] < 0.05 for w in valid_w)}/{len(valid_w)} unpenalized fits"
                       if valid_w else f"p n/a (all {len(ws)} fits penalized)")
            print(f"           DL score: train HR {np.mean([w[0] for w in ws]) if ws else float('nan'):.2f}"
                  f", {train_p}"
                  f"  |  held-out re-test HR {np.mean([t[0] for t in ts]) if ts else float('nan'):.2f}"
                  f", p<0.05 in {sum(t[3] < 0.05 for t in ts)}/{len(ts)} seeds")

        _write_cell(rows, comps, cohort, endpoint, spec_name, "few")
    print(f"   [{cohort}/{ENDPOINT_PAPER_NAME[endpoint]}: {'+'.join(regimes)} done in "
          f"{time.time()-t0:.0f}s]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", choices=["primary", "ajcc"], default="primary")
    ap.add_argument("--cohort", choices=list(COHORTS), default=None)
    ap.add_argument("--endpoint", choices=ENDPOINTS, default=None)
    ap.add_argument("--force", action="store_true", help="recompute cells already on disk")
    ap.add_argument("--regime", choices=["all", "full", "few"], default="all",
                    help="full = full-shot only, few = few-shot only, all = both (default)")
    ap.add_argument("--run", default=None,
                    help="run id YYYYMMDD_HHMMSS to resume / add cells to; default = new run")
    a = ap.parse_args()
    cols = SPEC_PRIMARY if a.spec == "primary" else SPEC_AJCC

    global RESULTS, CELLS
    run_id = a.run or new_run_id()
    RESULTS = results_dir("results", run_id)
    CELLS = RESULTS / "_cells"
    CELLS.mkdir(parents=True, exist_ok=True)
    tee_output(RESULTS / "logs" / f"{a.cohort or 'all'}_{a.endpoint or 'all'}_{a.regime}_{a.spec}.log")
    write_run_info(RESULTS, script="run_a1.py")
    print(f"command: {' '.join(sys.argv)}")
    print(f"imaging model: {IMAGING_LABEL[IMAGING]}   "
          f"{'features' if IMAGING == 'ctfm_st' else 'input'}: {FEATURE_VERSION}")
    print(f"N stage: {NSTAGE}" + ("   (AJCC, harmonised across cohorts)" if NSTAGE == "ajcc8"
          else "   (each cohort's own column -- different AJCC versions: NOT comparable)"))
    print(f"data version: {DATA_VERSION}   ({COHORTS['Radcure']['csv'].name}, "
          f"{COHORTS['Maastro']['csv'].name})")
    print(f"run {run_id}  ->  {RESULTS}")
    print(f"   (to resume or add cells to THIS run:  --run {run_id})")
    cohorts = [a.cohort] if a.cohort else list(COHORTS)
    endpoints = [a.endpoint] if a.endpoint else ENDPOINTS
    wanted = REGIMES if a.regime == "all" else (a.regime,)
    if a.force:
        (RESULTS / (".report_level" if a.spec == "primary" else f".report_level_{a.spec}")).unlink(missing_ok=True)
    for co in cohorts:
        for ep in endpoints:
            todo = [g for g in wanted if a.force or not _regime_done(co, ep, a.spec, g)]
            for g in wanted:
                if g not in todo:
                    print(f"[{co}/{ep}] {g}-shot already done  (use --force to redo)")
            if todo:
                run_cell(co, ep, a.spec, cols, tuple(todo))
    _maybe_build_report(a.spec, run_id)


if __name__ == "__main__":
    main()
