"""SegRes-E2E, LRF and DM: survival probabilities with the SUBMITTED paper's refit step (inference only, no training).

Why: in the submitted pipeline (working_space/few_shot_single_institution_pytorch_head_*.py) the saved full-shot
SegRes-E2E model used for calibration and IBS was not the trained linear head. After training, the script took the
512 pooled backbone features of the training patients, standardized them (StandardScaler) and refitted a lifelines
ridge Cox model (penalizer 1.0, l1_ratio 0); the bundle backbone + scaler + Cox coefficients produced the predictions.
The revision's OS/RFS SegRes-E2E calibration/IBS come from those submitted bundles (<outputs.dir>/paper_networks_ibs), but
LRF/DM come from the re-trained networks' own linear head, whose output is nearly constant -> vertical calibration.
This script applies the submitted refit step to the re-trained LRF/DM networks so all four endpoints are computed alike.

For each cohort (Radcure, Maastro), endpoint (lr, dr) and seed 0-4, the full-shot network of the main run
(outputs.paper_run or the newest; published: 20261104_054739; corrected events, the paper's input and loss) is loaded
unchanged; its pooled features
are computed for the training and test patients on the same cached scans it was trained on; scaler + ridge Cox are
fitted on the training patients; test survival uses the submitted calibration code (Breslow baseline on the training
risks, 2-year survival, IPCW IBS on the submitted time grid; working_space/generate_calibration_pytorch_head.py, the
same functions score_paper_networks_ibs.py used for OS/RFS -- copied verbatim into submitted_calibration.py, so this
step does not need the original code base). Five seeds -> the revision convention (make_calibration.py):
IBS = mean over seeds (recomputed in each of 2,000 bootstrap resamples), calibration error = mean over seeds,
calibration bins from the seed-averaged 2-year survival.

    python3 refit_segres_lrf_dm.py --dry-run          (CPU: files, labels, split alignment; no GPU work)
    CUDA_VISIBLE_DEVICES=3 python3 refit_segres_lrf_dm.py
Writes <outputs.dir>/segres_refit_lrf_dm/: summary.csv, bins.csv, per_seed.csv,
refit_<cohort>_<ep>_SegRes-E2E.npz (same fields as <outputs.dir>/paper_networks_ibs/*.npz; no patient IDs).
Existing pipeline code and outputs are not changed.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ctfm_config import outputs_dir, paper_run    # noqa: E402

RUN = paper_run(must_exist=False)[0]          # outputs.paper_run, or the newest run (checked in main)
NETS = RUN / "segres_e2e" / "corrected_published_fp32"
OUT = outputs_dir() / "segres_refit_lrf_dm"
FULL_N = {"Radcure": 940, "Maastro": 121}
SEEDS = range(5)
EPS = {"lr": "LRF", "dr": "DM"}
PENALIZER, L1_RATIO = 1.0, 0.0          # submitted cfg.regularizer {"name": "Ridge_1.0", ...}


def cindex(t, e, r):
    from lifelines.utils import concordance_index
    return float(concordance_index(t, -np.asarray(r), e))


def labels_and_split(co, ep, j, keys):
    """Training / test rows of a network: main.npz stores its own cohort in the order train, val, test."""
    import cohort
    df = cohort.build_cohort(co, ep).set_index("key")
    n_tr, n_va = j["n_train"], j["n_val"]
    k_tr, k_te = list(keys[:n_tr]), list(keys[n_tr + n_va:])
    t = lambda ks: df.loc[ks, "time"].values.astype(float)
    e = lambda ks: df.loc[ks, "event"].values.astype(float)
    return k_tr, t(k_tr), e(k_tr), k_te, t(k_te), e(k_te)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dry-run", action="store_true", help="check inputs only (CPU), no GPU work")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--cohorts", nargs="+", default=["Radcure", "Maastro"])
    a = ap.parse_args()

    paper_run()                                   # stops with a clear message when there is no run
    sys.path.insert(0, str(HERE))
    import runner as PR
    jobs = []
    for co in a.cohorts:
        for ep in EPS:
            for s in SEEDS:
                d = NETS / f"{co}_{ep}_n{FULL_N[co]}_s{s}"
                for f in ("main.pt", "main.json", "main.npz"):
                    if not (d / f).exists():
                        raise SystemExit(f"missing {d / f}")
                jobs.append((co, ep, s, d))
    print(f"{len(jobs)} networks found under {NETS}", flush=True)

    if a.dry_run:
        # setup per input once (lr -> lp2, dr -> lp); check labels and split by reproducing each network's test C-index
        for ep in EPS:
            PR.setup("corrected", None, None, segres_input=PR.LP_INPUT[ep])
            for co, e_, s, d in [x for x in jobs if x[1] == ep]:
                j = json.loads((d / "main.json").read_text())
                z = np.load(d / "main.npz", allow_pickle=True)
                keys, risk = z[f"keys_{co}"], z[f"risk_{co}"]
                k_tr, t_tr, e_tr, k_te, t_te, e_te = labels_and_split(co, ep, j, keys)
                c = cindex(t_te, e_te, risk[len(keys) - len(k_te):])
                ok = abs(c - j["test_c"]) < 1e-6 and int(e_tr.sum()) == j["events_train"]
                print(f"  {co:7} {EPS[ep]:3} s{s}: train {len(k_tr)} ({int(e_tr.sum())} ev)  test {len(k_te)} "
                      f"({int(e_te.sum())} ev)  head C {c:.4f} vs saved {j['test_c']:.4f}  "
                      f"{'OK' if ok else 'MISMATCH'}", flush=True)
                if not ok:
                    raise SystemExit("labels/split do not reproduce the saved network -- stop")
        print("dry run OK: no GPU work done", flush=True)
        return

    import torch
    import torch.nn.functional as F
    from lifelines import CoxPHFitter
    from sklearn.preprocessing import StandardScaler
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    per_seed, summary, bins = [], [], []
    for ep in EPS:
        cohort_mod, TS = PR.setup("corrected", None, None, segres_input=PR.LP_INPUT[ep])
        import submitted_calibration as SP        # submitted calibration code (G), km_bins, bootstrap, IBS (verbatim)
        G = SP.G
        cfg = G.Config()
        for co in a.cohorts:
            vol = TS.Volumes(co, device)
            S730, SURV, R_REFIT, C_HEAD, C_REFIT, CE = [], [], [], [], [], []
            et = None
            for _, _, s, d in [x for x in jobs if x[0] == co and x[1] == ep]:
                t0 = time.time()
                j = json.loads((d / "main.json").read_text())
                z = np.load(d / "main.npz", allow_pickle=True)
                keys, head = z[f"keys_{co}"], z[f"risk_{co}"]
                k_tr, t_tr, e_tr, k_te, t_te, e_te = labels_and_split(co, ep, j, keys)
                model = TS.build_model().to(device)
                model.load_state_dict(torch.load(d / "main.pt", map_location=device))
                model.eval()

                @torch.no_grad()
                def feats(ks):
                    rows, out = vol.rows(ks), []
                    for b in range(0, len(rows), 8):
                        f = model.backbone(vol.get(rows[b:b + 8]))[-1]
                        out.append(F.adaptive_avg_pool3d(f, 1).flatten(1).float().cpu().numpy())
                    return np.concatenate(out)
                f_tr, f_te = feats(k_tr), feats(k_te)
                # sanity: the loaded network reproduces its own saved head output
                with torch.no_grad():
                    h_te = model.head(torch.as_tensor(f_te, device=device)).cpu().numpy().ravel()
                head_te = head[len(keys) - len(k_te):]
                head_diff = float(np.max(np.abs(h_te - head_te)))
                # the submitted refit: StandardScaler + lifelines ridge Cox on the training features
                sc = StandardScaler().fit(f_tr)
                cols = [f"f{i}" for i in range(f_tr.shape[1])]
                dtr = pd.DataFrame(sc.transform(f_tr), columns=cols)
                dtr["time"], dtr["event"] = t_tr, e_tr
                cph = CoxPHFitter(penalizer=PENALIZER, l1_ratio=L1_RATIO)
                try:
                    cph.fit(dtr, duration_col="time", event_col="event", show_progress=False)
                    fit_note = "ok"
                except Exception as ex:                 # retry with a smaller Newton step, as lifelines suggests
                    cph = CoxPHFitter(penalizer=PENALIZER, l1_ratio=L1_RATIO)
                    cph.fit(dtr, duration_col="time", event_col="event", show_progress=False,
                            fit_options={"step_size": 0.1})
                    fit_note = f"retried step 0.1 ({type(ex).__name__})"
                beta = cph.params_.values
                r_tr = sc.transform(f_tr) @ beta
                r_te = sc.transform(f_te) @ beta
                # submitted calibration pipeline: Breslow on the training risks, IPCW IBS over the clinical grid
                bt, bs, centre = G.breslow_baseline_survival(t_tr, e_tr, r_tr)
                tmax = bt[-1] if len(bt) else t_tr.max()
                et_s = cfg.eval_times[cfg.eval_times <= tmax * 0.95]
                et = et_s if et is None else et
                assert np.array_equal(et, et_s), "evaluation grid differs between seeds"
                surv = G.risk_to_survival_probs(r_te, bt, bs, et, centre)
                ibs_g = G.compute_brier_score_ipcw(surv, et, t_te, e_te)[2]
                ibs_v = SP.ibs_from_probs(surv, t_te, e_te, et)
                assert abs(ibs_g - ibs_v) < 1e-6, f"IBS functions disagree: {ibs_g} vs {ibs_v}"
                s730 = G.risk_to_survival_probs(r_te, bt, bs, np.array([730.0]), centre).flatten()
                b = SP.km_bins(s730, t_te, e_te, n_bins=5 if len(t_te) >= 100 else 3)
                ce = float(np.mean([abs(x["predicted"] - x["observed"]) for x in b])) if b else np.nan
                c_head, c_ref = cindex(t_te, e_te, head_te), cindex(t_te, e_te, r_te)
                S730.append(s730); SURV.append(surv); R_REFIT.append(r_te)
                C_HEAD.append(c_head); C_REFIT.append(c_ref); CE.append(ce)
                per_seed.append({"cohort": co, "endpoint": EPS[ep], "seed": s, "n_train": len(k_tr),
                                 "events_train": int(e_tr.sum()), "n_test": len(k_te), "events_test": int(e_te.sum()),
                                 "cindex_head": c_head, "cindex_refit": c_ref, "ibs_refit": ibs_g, "calib_error_refit": ce,
                                 "s730_min": float(s730.min()), "s730_max": float(s730.max()),
                                 "head_output_sd_train": float(np.std(head[:len(k_tr)])),
                                 "head_reproduction_maxdiff": head_diff, "cox_fit": fit_note})
                print(f"  {co:7} {EPS[ep]:3} s{s}: head C {c_head:.3f}  refit C {c_ref:.3f}  IBS {ibs_g:.4f}  "
                      f"CE {ce:.3f}  2-y surv {s730.min():.3f}-{s730.max():.3f}  head diff {head_diff:.1e}  "
                      f"cox {fit_note}  ({time.time() - t0:.0f}s)", flush=True)
                del model
                torch.cuda.empty_cache()
            # five seeds -> revision convention
            SURV, S730 = np.asarray(SURV), np.asarray(S730)
            fn = lambda i: float(np.mean([SP.ibs_from_probs(sv[i], t_te[i], e_te[i], et) for sv in SURV]))
            ibs, lo, hi, _ = SP.bootstrap_ci(fn, t_te, e_te, B=a.boot)
            pm = S730.mean(0)
            bm = SP.km_bins(pm, t_te, e_te, n_bins=5 if len(t_te) >= 100 else 3)
            summary.append({"variant": "refit", "cohort": co, "endpoint": EPS[ep], "model": "SegRes-E2E",
                            "n_test": len(t_te), "n_events": int(e_te.sum()), "n_seeds": len(SURV),
                            "ibs": ibs, "ibs_lo": lo, "ibs_hi": hi, "calib_error": float(np.mean(CE)),
                            "cindex_head_mean": float(np.mean(C_HEAD)), "cindex_refit_mean": float(np.mean(C_REFIT)),
                            "s730_seedmean_min": float(pm.min()), "s730_seedmean_max": float(pm.max())})
            bins += [{"variant": "refit", "cohort": co, "endpoint": EPS[ep], "model": "SegRes-E2E", **x} for x in bm]
            np.savez(OUT / f"refit_{co}_{ep}_SegRes-E2E.npz", test_times=t_te, test_events=e_te,
                     test_risk=np.asarray(R_REFIT).mean(0), surv=SURV.mean(0), eval_times=et, surv730=pm,
                     surv730_per_seed=S730)
            print(f"== {co} {EPS[ep]}: IBS {ibs:.4f} [{lo:.4f}-{hi:.4f}]  CE {np.mean(CE):.3f}  "
                  f"seed-mean 2-y surv {pm.min():.3f}-{pm.max():.3f}  bins pred "
                  f"{[round(x['predicted'], 3) for x in bm]}  obs {[round(x['observed'], 3) for x in bm]}", flush=True)
            pd.DataFrame(per_seed).to_csv(OUT / "per_seed.csv", index=False)
            pd.DataFrame(summary).to_csv(OUT / "summary.csv", index=False)
            pd.DataFrame(bins).to_csv(OUT / "bins.csv", index=False)
            del vol
            torch.cuda.empty_cache()
    print(f"done -> {OUT / 'summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
