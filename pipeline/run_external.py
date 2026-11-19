"""External (cross-cohort) validation -- reviewer items E1, R2-7, R3-8.

Trains on one cohort and evaluates on EVERY patient of the other, which is what
R2 asked for: "build the AI model on the RadCure data and independently evaluate
on the Maastricht data ... a much more compelling demonstration than the random
train/validation/test split you did."

Three arms, all scored on the identical target cohort:
    clinical_only   age, sex, HPV, smoking, T_Stage, N_Stage
    ctfm_st         512 frozen CT-FM features, ridge Cox
    combined_A      6 clinical + 1 out-of-fold imaging risk score

With A1_IMAGING=segres_ft the imaging arm is segres_ft (SegResEncoder trained from
scratch on the source cohort by train_segres.py); everything else is identical.

Everything the target cohort needs -- the feature scaler, the age standardisation,
the ridge penalty, the Cox coefficients -- is derived on the SOURCE cohort only.
The target is touched once, at scoring time.

    python run_external.py                             # Radcure -> Maastro, refit
    python run_external.py --source-model bundle       # reuse the published bundles
    python run_external.py --direction maastro_to_radcure
    python run_external.py --train-on all              # fit on the full source cohort

Run extract_features.py once first.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test

import bundles
from bootstrap_ci import bootstrap_ci, cindex, paired_bootstrap
from cohort import (DATA_VERSION, FEATURE_VERSION, IMAGING, IMAGING_LABEL, IMAGING_SHORT,
                    NSTAGE, new_run_id, results_dir, tee_output, write_run_info, ENDPOINTS,
                    ENDPOINT_PAPER_NAME, SPEC_AJCC, SPEC_PRIMARY, build_cohort,
                    split_dataset_ratio)
from extract_features import load_features
from ipcw_brier import eval_times_for, ibs_from_probs
from models import (dl_score_independent_test, dl_score_wald,
                    fit_clinical_and_combined, fit_ctfm_st, fit_segres_ft)

OUT = None            # set below: results_external_<version>_<run id>/
ARMS = ["clinical_only", IMAGING, "combined_A"]          # IMAGING: ctfm_st or segres_ft
B_BOOT = 2000


def km_tertiles(times, events, risk, horizon=730.0):
    """Split the target cohort into risk tertiles; KM survival + log-rank (R2-6)."""
    grp = np.digitize(risk, np.quantile(risk, [1 / 3, 2 / 3]))
    rows = []
    for g, name in enumerate(["low", "medium", "high"]):
        m = grp == g
        if m.sum() < 3:
            continue
        kmf = KaplanMeierFitter().fit(times[m], events[m])
        surv = (float(kmf.survival_function_at_times(horizon).iloc[0])
                if horizon <= kmf.survival_function_.index.max()
                else float(kmf.survival_function_.iloc[-1].iloc[0]))
        rows.append({"tertile": name, "n": int(m.sum()),
                     "events": int(events[m].sum()), "surv_2yr": surv})
    try:
        lr = multivariate_logrank_test(times, grp, events)
        stat, p = float(lr.test_statistic), float(lr.p_value)
    except Exception:
        stat, p = np.nan, np.nan
    return rows, stat, p, grp


def run(direction: str, source_model: str, train_on: str, spec_name: str, cols: list[str],
        endpoints=None):
    src, tgt = ("Radcure", "Maastro") if direction == "radcure_to_maastro" else ("Maastro", "Radcure")
    endpoints = endpoints or ENDPOINTS
    if IMAGING == "segres_ft":
        if train_on != "train" or source_model != "refit":
            raise SystemExit("SegRes-FT: only --train-on train --source-model refit "
                             "(its networks are trained on the source training split)")
        import segres_cache
        segres_cache.check_trained([src], endpoints, ("full",))
        f_src_map = f_tgt_map = None              # no features: networks' outputs instead
    else:
        f_src_map, f_tgt_map = load_features(src), load_features(tgt)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "predictions").mkdir(exist_ok=True)
    rows, comps, kms = [], [], []

    print(f"data version: {DATA_VERSION}   imaging model: {IMAGING_LABEL[IMAGING]}   "
          f"{'features' if IMAGING == 'ctfm_st' else 'input'}: {FEATURE_VERSION}   "
          f"N stage: {NSTAGE}   results -> {OUT}")
    print(f"\n{'='*72}\nEXTERNAL VALIDATION   {src} -> {tgt}"
          f"   (imaging: {IMAGING_LABEL[IMAGING]} {source_model}, train-on: {train_on})\n{'='*72}")

    for endpoint in endpoints:
        t0 = time.time()
        d_src, d_tgt = build_cohort(src, endpoint), build_cohort(tgt, endpoint)
        tr_split, va, _ = split_dataset_ratio(d_src)
        train = d_src if train_on == "all" else tr_split

        get = lambda d, m: None if m is None else np.stack([m[k] for k in d["key"]])
        f_tr, f_va, f_tgt = get(train, f_src_map), get(va, f_src_map), get(d_tgt, f_tgt_map)
        eval_times = eval_times_for(train["time"].values, train["event"].values)
        times, events = d_tgt["time"].values, d_tgt["event"].values

        base = dict(direction=f"{src}->{tgt}", source=src, target=tgt,
                    endpoint=ENDPOINT_PAPER_NAME[endpoint], endpoint_code=endpoint,
                    spec=spec_name, imaging_source=source_model, train_on=train_on,
                    n_train=len(train), n_target=len(d_tgt),
                    n_events=int(events.sum()))
        print(f"\n[{ENDPOINT_PAPER_NAME[endpoint]}] train {src} n={len(train)} "
              f"-> test {tgt} n={len(d_tgt)} ({int(events.sum())} events)")

        # With --train-on all the val split sits INSIDE `train`, so selecting lambda on
        # it would be in-sample. Select lambda on the held-out train/val split first,
        # then refit on the full cohort with that fixed lambda.
        stage1 = None
        if IMAGING == "segres_ft":            # the network of the source's full training split
            imaging, stage1 = fit_segres_ft(train, va, d_tgt, eval_times, src, endpoint, 42,
                                            test_cohort=tgt)
        elif train_on == "all":
            probe = fit_ctfm_st(tr_split, va, d_tgt, get(tr_split, f_src_map),
                                f_va, f_tgt, eval_times)
            lam = (float(probe.hyperparam.split("=")[1])
                   if "ridge=" in probe.hyperparam else 1.0)
            imaging = fit_ctfm_st(train, va, d_tgt, f_tr, f_va, f_tgt, eval_times,
                                  fixed_lambda=lam)
        else:
            imaging = fit_ctfm_st(train, va, d_tgt, f_tr, f_va, f_tgt, eval_times)
        clin, comb = fit_clinical_and_combined(train, va, d_tgt, f_tr, f_va, f_tgt, cols,
                                               eval_times, imaging, 42, stage1=stage1)
        fits = {"clinical_only": clin, IMAGING: imaging, "combined_A": comb}

        # optional cross-check: score the target with the PUBLISHED bundle instead
        if source_model == "bundle":
            try:
                risk_b = bundles.bundle_risk(f_tgt, endpoint, src)
                fits["ctfm_st"].risk_test = risk_b
                fits["ctfm_st"].surv_test = None      # bundle has no baseline hazard -> C-index only
                fits["ctfm_st"].hyperparam = f"published bundle ({src})"
                fits["clinical_only"], fits["combined_A"] = fit_clinical_and_combined(
                    train, va, d_tgt, f_tr, f_va, f_tgt, cols, eval_times, fits["ctfm_st"], 42)
            except ValueError as e:
                print(f"   !! bundle unusable for {endpoint}: {e}")
                print("      falling back to refit for this endpoint")

        for arm, res in fits.items():
            ok = (not res.failed) and np.isfinite(res.risk_test).all()
            m = {"cindex": np.nan, "cindex_lo": np.nan, "cindex_hi": np.nan,
                 "ibs": np.nan, "ibs_lo": np.nan, "ibs_hi": np.nan}
            if ok:
                p, lo, hi, _ = bootstrap_ci(
                    lambda i: cindex(times[i], events[i], res.risk_test[i]),
                    times, events, B=B_BOOT)
                m.update(cindex=p, cindex_lo=lo, cindex_hi=hi)
                if res.surv_test is not None and np.isfinite(res.surv_test).all():
                    sv = res.surv_test
                    p, lo, hi, _ = bootstrap_ci(
                        lambda i: ibs_from_probs(sv[i], times[i], events[i], eval_times),
                        times, events, B=B_BOOT)
                    m.update(ibs=p, ibs_lo=lo, ibs_hi=hi)
            rows.append({**base, "arm": arm, "epv": res.epv, "hyperparam": res.hyperparam,
                         "failed": res.failed, **m})
            print(f"   {arm:<14} C={m['cindex']:.4f} [{m['cindex_lo']:.3f},{m['cindex_hi']:.3f}]"
                  f"  IBS={m['ibs']:.4f}  {res.hyperparam}"
                  + (f"  FAILED: {res.failed}" if res.failed else ""))

            if ok:
                tert, stat, pv, grp = km_tertiles(times, events, res.risk_test)
                for t in tert:
                    kms.append({**base, "arm": arm, **t,
                                "logrank_chi2": stat, "logrank_p": pv})
                if tert:
                    print("      KM tertiles 2-yr survival: " +
                          ", ".join(f"{t['tertile']}={t['surv_2yr']:.3f}(n={t['n']})"
                                    for t in tert) +
                          f"   log-rank p={pv:.3g}")
                np.savez_compressed(
                    OUT / "predictions" /
                    f"{src}_to_{tgt}_{endpoint}_{arm}_{source_model}_{train_on}_{spec_name}.npz",
                    keys=d_tgt["key"].values, times=times, events=events,
                    risk=res.risk_test, tertile=grp, eval_times=eval_times,
                    surv=res.surv_test if res.surv_test is not None else np.empty((0, 0)))

        good = {a: f for a, f in fits.items()
                if not f.failed and np.isfinite(f.risk_test).all()}
        for a, b in [("combined_A", "clinical_only"), (IMAGING, "clinical_only"),
                     ("combined_A", IMAGING)]:
            if a in good and b in good:
                ra, rb = good[a].risk_test, good[b].risk_test
                dd, lo, hi, p = paired_bootstrap(
                    lambda i: cindex(times[i], events[i], ra[i]),
                    lambda i: cindex(times[i], events[i], rb[i]),
                    times, events, B=B_BOOT)
                comps.append({**base, "metric": "cindex", "model_a": a, "model_b": b,
                              "delta": dd, "lo": lo, "hi": hi, "p": p})
                print(f"   Δ {a} − {b}: {dd:+.4f} [{lo:+.3f},{hi:+.3f}] p={p:.4f}")

        # ---- DL-score significance: (1) Wald inside the combined model, training data;
        #      (2) frozen clinical + DL scores re-tested on patients neither was fitted on.
        w = dl_score_wald(fits.get("combined_A"))
        if w is not None:
            comps.append({**base, "metric": "dl_wald_train", "model_a": "combined_A",
                          "model_b": "", "delta": w[0], "lo": w[1], "hi": w[2], "p": w[3],
                          "penalized": w[4]})
            print(f"   DL score in combined model (train): HR/SD {w[0]:.3f} [{w[1]:.2f},{w[2]:.2f}]"
                  f" p={w[3]:.3g}" + ("   (penalized fit: p not valid)" if w[4] else ""))
        cl, im = fits.get("clinical_only"), fits.get(IMAGING)
        if cl is not None and im is not None and not cl.failed and not im.failed:
            t = dl_score_independent_test(times, events, cl.risk_test, im.risk_test)
            if t is not None:
                comps.append({**base, "metric": "dl_independent_test",
                              "model_a": "dl_score", "model_b": "clinical_score",
                              "delta": t[0], "lo": t[1], "hi": t[2], "p": t[3]})
                print(f"   DL score re-tested on external cohort: HR/SD {t[0]:.3f} [{t[1]:.2f},{t[2]:.2f}] p={t[3]:.3g}")

        print(f"   [{time.time()-t0:.0f}s]")

    tag = f"{src}_to_{tgt}_{source_model}_{train_on}_{spec_name}"
    pd.DataFrame(rows).to_csv(OUT / f"external_results_{tag}.csv", index=False)
    pd.DataFrame(comps).to_csv(OUT / f"external_comparisons_{tag}.csv", index=False)
    pd.DataFrame(kms).to_csv(OUT / f"external_km_tertiles_{tag}.csv", index=False)
    summary = _summary_md(pd.DataFrame(rows), pd.DataFrame(comps), pd.DataFrame(kms), src, tgt,
                          source_model, train_on)
    (OUT / f"external_summary_{tag}.md").write_text(summary)
    print("\n" + summary)
    print(f"wrote -> {OUT}/external_summary_{tag}.md  (+ results / comparisons / km_tertiles CSVs)")
    try:
        import make_external_report
        make_external_report.OUT = OUT
        make_external_report.main(tag)
    except Exception as exc:                       # results are already on disk
        print(f"!! figures not made ({exc}); re-run: python make_external_report.py --tag {tag}")


def _summary_md(rows, comps, kms, src, tgt, source_model, train_on):
    """One readable table: every endpoint x model on the external cohort."""
    LABEL = {"clinical_only": "Clinical only", IMAGING: IMAGING_LABEL[IMAGING],
             "combined_A": f"Clinical + {IMAGING_SHORT[IMAGING]}"}
    def ci(r, m):
        if not np.isfinite(r[m]):
            return "—"
        lo, hi = r[f"{m}_lo"], r[f"{m}_hi"]
        return f"{r[m]:.3f} ({lo:.3f}–{hi:.3f})" if np.isfinite(lo) else f"{r[m]:.3f}"
    def comp(ep, metric, a=None, b=None):
        if comps.empty:
            return None
        g = comps[(comps.endpoint == ep) & (comps.metric == metric)]
        if a is not None:
            g = g[g.model_a == a]
        if b is not None:
            g = g[g.model_b == b]
        return None if g.empty else g.iloc[0]
    lines = [f"## External validation: train {src} -> test all {tgt}",
             "",
             f"Imaging model: {IMAGING_LABEL[IMAGING]} ({source_model}). Source training set: {train_on}. "
             f"Data version: {DATA_VERSION}. "
             + ("N stage harmonised to AJCC (p16-specific) in both cohorts. "
                if NSTAGE == "ajcc8" else
                "N stage as each cohort recorded it (different AJCC versions) -- "
                "NOT on a common scale, so the clinical models transfer across different scales. ")
             + f"95% CIs: {B_BOOT:,} event-stratified bootstrap "
             f"resamples of the {tgt} cohort. Nothing from {tgt} is used for fitting.",
             "",
             "| Endpoint | Events | Model | C-index (95% CI) | IBS (95% CI) | ΔC vs clinical (p) | KM tertiles 2-yr survival (log-rank p) |",
             "|---|---|---|---|---|---|---|"]
    for ep in [ENDPOINT_PAPER_NAME[e] for e in ENDPOINTS]:
        sub = rows[rows.endpoint == ep]
        for arm in ARMS:
            r = sub[sub.arm == arm]
            if r.empty:
                continue
            r = r.iloc[0]
            d = comp(ep, "cindex", arm, "clinical_only") if arm != "clinical_only" else None
            dtxt = "—" if d is None else f"{d.delta:+.3f} (p={d.p:.3f})"
            k = kms[(kms.endpoint == ep) & (kms.arm == arm)] if not kms.empty else kms
            ktxt = ("—" if k.empty else
                    " / ".join(f"{x.surv_2yr:.2f}" for x in k.itertuples())
                    + f" (p={k.iloc[0].logrank_p:.2g})")
            lines.append(f"| {ep} | {int(r.n_events)} | {LABEL[arm]} | {ci(r, 'cindex')} | "
                         f"{ci(r, 'ibs')} | {dtxt} | {ktxt} |")
    lines += ["", "### DL score contribution (combined model)", "",
              f"| Endpoint | HR/SD in combined model, {src} training (p) | "
              f"HR/SD re-tested on {tgt} (p) |", "|---|---|---|"]
    for ep in [ENDPOINT_PAPER_NAME[e] for e in ENDPOINTS]:
        w, t = comp(ep, "dl_wald_train"), comp(ep, "dl_independent_test")
        wt = "—" if w is None else (f"{w.delta:.2f} ({w.lo:.2f}–{w.hi:.2f}), p={w.p:.2g}"
                                    + (" † penalized" if bool(w.get("penalized", False)) else ""))
        tt = "—" if t is None else f"{t.delta:.2f} ({t.lo:.2f}–{t.hi:.2f}), p={t.p:.2g}"
        lines.append(f"| {ep} | {wt} | {tt} |")
    lines += ["", "KM tertiles = low / medium / high predicted risk thirds of the external cohort."]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--direction", default="radcure_to_maastro",
                    choices=["radcure_to_maastro", "maastro_to_radcure"])
    ap.add_argument("--source-model", default="refit", choices=["refit", "bundle"],
                    help="refit CT-FM-ST on the source (default), or load the published bundle")
    ap.add_argument("--train-on", default="train", choices=["train", "all"],
                    help="source training split (default, matches the bundles) or full cohort")
    ap.add_argument("--spec", default="primary", choices=["primary", "ajcc"])
    ap.add_argument("--endpoint", default=None, choices=ENDPOINTS,
                    help="one endpoint only, for quick checks (default: all four)")
    ap.add_argument("--run", default=None,
                    help="run id YYYYMMDD_HHMMSS to write into; default = new run")
    a = ap.parse_args()
    run_id = a.run or new_run_id()
    OUT = results_dir("results_external", run_id)
    tee_output(OUT / "logs" / f"{a.direction}_{a.source_model}_{a.train_on}_{a.spec}.log")
    write_run_info(OUT, script="run_external.py")
    print(f"command: {' '.join(sys.argv)}")
    print(f"run {run_id}  ->  {OUT}")
    run(a.direction, a.source_model, a.train_on, a.spec,
        SPEC_PRIMARY if a.spec == "primary" else SPEC_AJCC,
        [a.endpoint] if a.endpoint else None)
