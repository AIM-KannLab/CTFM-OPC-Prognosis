"""Calibration at 2 years -- Table S6 / Figure 5 of the manuscript, rebuilt with 95% CIs.

Replaces working_space/generate_calibration_from_saved_model.py, which scored saved
bundles; the OS bundles carry an identity scaler (mean 0, SD 1) although their Cox
coefficients were fitted on standardized features, so the submitted OS calibration rows
are wrong. This script reads the survival curves each run saved for its own test patients
(full shot), so nothing is refitted or re-extracted.

Per model x cohort x endpoint, on the held-out test set:
  IBS                      integrated Brier score (IPCW, same code as everywhere else)
  calibration slope        Cox regression of the outcome on the model's risk score in the
                           test patients: 1 = well calibrated, < 1 = predictions too
                           extreme (overfitting), > 1 = too timid
  calibration-in-the-large mean predicted minus Kaplan-Meier observed 2-year survival over
                           the whole test set (negative = the model is too pessimistic)
each with a 95% CI from 2,000 event-stratified bootstrap resamples of the test patients, and
  calibration error        the submitted definition, kept for comparability: mean absolute
                           gap between predicted and observed 2-year survival over bins of
                           predicted risk (quintiles; tertiles for the 26-patient Maastro
                           test set). POINT ESTIMATE ONLY: an absolute deviation is inflated
                           by resampling noise, so its bootstrap distribution sits above the
                           estimate and a percentile CI would not even contain it.
Models fitted with 5 seeds (CT-FM-LP, SegRes-E2E) get the mean over seeds, recomputed in
every resample -- the same method as make_table_s4.py.

Observed survival is Kaplan-Meier. For LR and DM the models predict death-censored
(cause-specific) risk, so the observed side censors death too: like is compared with like.

SPECTRE is left out, as in the submitted Figure 5.

    python make_calibration.py --runs <run> [<run2> ...] [--out <folder>]
"""
from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from bootstrap_ci import bootstrap_ci
from cohort import ENDPOINT_PAPER_NAME, ENDPOINTS
from ipcw_brier import ibs_from_probs
from make_table_s4 import LABEL, ORDER, _arms, _run_info, same_data_or_exit

HORIZON = 730.0                  # days: 2 years, as in the submitted Figure 5
COHORTS = ["Radcure", "Maastro"]
B_BOOT = 2000
EXCLUDE = {"spectre"}            # not in the submitted Figure 5


def km_at(t, e, h=HORIZON) -> float:
    """Kaplan-Meier survival at time h."""
    s = 1.0
    for u in np.unique(t[(e == 1) & (t <= h)]):
        s *= 1.0 - ((t == u) & (e == 1)).sum() / (t >= u).sum()
    return float(s)


def _bins(pred, n_bins):
    """Equal-size bins of patients ordered by predicted survival (robust to ties)."""
    return np.array_split(np.argsort(pred, kind="stable"), n_bins)


def cox_slope(lp, t, e, iters=30) -> float:
    """Calibration slope: the coefficient of a one-covariate Cox model of the outcome on the
    risk score lp (Breslow ties), by Newton-Raphson. Raises ValueError if it does not
    converge, which the bootstrap counts as an unusable resample."""
    o = np.argsort(-t, kind="stable")
    lp, t, e = lp[o], t[o], e[o]
    last = np.searchsorted(-t, -t, side="right") - 1      # end of each tie group (risk set)
    ev = e == 1
    b = 0.0
    for _ in range(iters):
        x = b * lp
        w = np.exp(x - x.max())
        S0, S1, S2 = np.cumsum(w)[last], np.cumsum(w * lp)[last], np.cumsum(w * lp * lp)[last]
        m = S1[ev] / S0[ev]
        U = float((lp[ev] - m).sum())
        info = float((S2[ev] / S0[ev] - m * m).sum())
        if not np.isfinite(info) or info <= 1e-12:
            raise ValueError("no information about the slope")
        step = U / info
        b += step
        if abs(step) < 1e-8:
            return float(b)
        if abs(b) > 50:
            break
    raise ValueError("calibration slope did not converge")


def calib_error(pred, t, e, n_bins) -> float:
    return float(np.mean([abs(pred[b].mean() - km_at(t[b], e[b])) for b in _bins(pred, n_bins)]))


def _load(path: Path):
    z = np.load(path, allow_pickle=True)
    n0 = len(z["keys"]) if "keys" in z.files else None
    k = int(z["n_seeds"]) if "n_seeds" in z.files else 1
    n0 = n0 or len(z["times"]) // k
    t, e = z["times"].astype(float)[:n0], z["events"].astype(int)[:n0]
    if not z["surv"].size:
        return None
    S = z["surv"].astype(float).reshape(k, n0, -1)
    et = np.asarray(z["eval_times"], float)
    if HORIZON in et:
        P = S[:, :, int(np.flatnonzero(et == HORIZON)[0])]
    else:                                         # interpolate on the saved grid
        P = np.array([[np.interp(HORIZON, et, row) for row in s] for s in S])
    R = z["risk"].astype(float).reshape(k, n0)
    return t, e, S, P, et, R


def _cell(job):
    key, path = job
    got = _load(path)
    if got is None:
        return key, None
    t, e, S, P, et, R = got
    n_bins = 5 if len(t) >= 100 else 3
    ce = lambda i: float(np.mean([calib_error(p[i], t[i], e[i], n_bins) for p in P]))
    citl = lambda i: float(np.mean([p[i].mean() for p in P]) - km_at(t[i], e[i]))
    ibs = lambda i: float(np.mean([ibs_from_probs(s[i], t[i], e[i], et) for s in S]))
    slope = lambda i: float(np.mean([cox_slope(r[i], t[i], e[i]) for r in R]))
    out = {"n_test": len(t), "n_events": int(e.sum()), "n_seeds": len(P), "n_bins": n_bins,
           "observed_2yr": km_at(t, e), "predicted_2yr": float(P.mean()),
           "calib_error": ce(np.arange(len(t)))}
    for name, fn in [("ibs", ibs), ("slope", slope), ("citl", citl)]:
        p, lo, hi, _ = bootstrap_ci(fn, t, e, B=B_BOOT)
        out.update({name: p, f"{name}_lo": lo, f"{name}_hi": hi})
    # figure: bins of the seed-averaged prediction, observed KM with its 95% CI
    from lifelines import KaplanMeierFitter
    pm = P.mean(0)
    bins = []
    for b in _bins(pm, n_bins):
        km = KaplanMeierFitter().fit(t[b], e[b])
        ci = km.confidence_interval_survival_function_
        at = ci.index[ci.index <= HORIZON]
        lo_, hi_ = (ci.loc[at[-1]].values if len(at) else (1.0, 1.0))
        bins.append({"predicted": float(pm[b].mean()), "observed": km_at(t[b], e[b]),
                     "obs_lo": float(lo_), "obs_hi": float(hi_), "n": len(b)})
    out["bins"] = bins
    return key, out


def main(runs=None, spec="primary", out=None, procs=8):
    if runs is None:
        ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
        ap.add_argument("--runs", nargs="+", required=True)
        ap.add_argument("--spec", choices=["primary", "ajcc"], default="primary")
        ap.add_argument("--out", default=None)
        ap.add_argument("--procs", type=int, default=8)
        a = ap.parse_args()
        runs, spec, out, procs = a.runs, a.spec, a.out, a.procs
    runs = [Path(r).resolve() for r in runs]
    same_data_or_exit(runs)
    sources = {}
    for r in runs:
        for arm, pred in _arms(r, _run_info(r)["imaging"]).items():
            sources.setdefault(arm, pred)
    arms = [x for x in ORDER if x in sources and x not in EXCLUDE]
    print(f"calibration: models {', '.join(LABEL.get(x, x) for x in arms)}")
    jobs = []
    for co in COHORTS:
        for ep in ENDPOINTS:
            for arm in arms:
                f = sources[arm] / f"{co}_{ep}_{spec}_fullshot.npz"
                if not f.exists():                # imaging-only baselines use no clinical variable
                    f = sources[arm] / f"{co}_{ep}_primary_fullshot.npz"
                if f.exists():
                    jobs.append(((co, ENDPOINT_PAPER_NAME[ep], arm), f))
    print(f"{len(jobs)} cells, {B_BOOT:,} bootstrap resamples each ...", flush=True)
    with Pool(procs) as pool:
        res = dict(r for r in pool.imap_unordered(_cell, jobs) if r[1] is not None)

    out = Path(out) if out else runs[0] / ("_report" if spec == "primary" else f"_report_{spec}")
    (out / "figures").mkdir(parents=True, exist_ok=True)
    rows, bins = [], []
    for (co, ep, arm), r in sorted(res.items(), key=lambda kv: (COHORTS.index(kv[0][0]),
                                   ["OS", "RFS", "LR", "DM"].index(kv[0][1]), ORDER.index(kv[0][2]))):
        rows.append({"cohort": co, "endpoint": ep, "model": LABEL.get(arm, arm),
                     **{k: v for k, v in r.items() if k != "bins"}})
        bins += [{"cohort": co, "endpoint": ep, "model": LABEL.get(arm, arm), "bin": i, **b}
                 for i, b in enumerate(r["bins"])]
    L, B = pd.DataFrame(rows), pd.DataFrame(bins)
    L.to_csv(out / "table_s6_calibration_long.csv", index=False)
    B.to_csv(out / "fig5_calibration_bins.csv", index=False)

    f = lambda r, k, d=3: f"{r[k]:.{d}f} ({r[k + '_lo']:.{d}f}–{r[k + '_hi']:.{d}f})"
    W = pd.DataFrame([{"Cohort": r.cohort, "Endpoint": r.endpoint, "Model": r.model,
                       "IBS": f(r, "ibs"), "Calibration slope": f(r, "slope", 2),
                       "Predicted − observed 2-yr survival": f(r, "citl"),
                       "2-yr calibration error": f"{r.calib_error:.3f}",
                       "Observed 2-yr": f"{r.observed_2yr:.3f}"} for _, r in L.iterrows()])
    W.to_csv(out / "table_s6_calibration.csv", index=False)
    cols = list(W.columns)
    w = [max(len(str(c)), *(len(str(v)) for v in W[c])) for c in cols]
    info = _run_info(runs[0])
    note = (f"Held-out test sets, full-shot models. 95% CIs: {B_BOOT:,} event-stratified bootstrap "
            f"resamples of the test patients; models fitted with 5 seeds report the mean over "
            f"seeds. Calibration slope: Cox regression of the outcome on the model's risk score; "
            f"1 = ideal, < 1 = predictions too extreme. Predicted − observed: whole test set; "
            f"negative = model too pessimistic. Calibration error (as in the submitted table): mean "
            f"absolute difference between predicted and Kaplan-Meier observed 2-year survival over "
            f"bins of predicted risk (quintiles; tertiles for Maastro, n=26); point estimate only, "
            f"since resampling inflates an absolute deviation. LR and DM: death censored on both "
            f"sides. Data: {info['data']}; "
            f"input: {info['features']}; N stage: {info['nstage']}. "
            f"Runs: {', '.join(r.name for r in runs)}.")
    md = ["## Table S6 (revised). Calibration at 2 years, with 95% CI", "", note, "",
          "| " + " | ".join(str(c).ljust(w[k]) for k, c in enumerate(cols)) + " |",
          "|" + "|".join("-" * (x + 2) for x in w) + "|"]
    md += ["| " + " | ".join(str(r[c]).ljust(w[k]) for k, c in enumerate(cols)) + " |"
           for _, r in W.iterrows()]
    (out / "table_s6_calibration.md").write_text("\n".join(md) + "\n")
    print(f"\nwrote {out / 'table_s6_calibration.md'}")
    fig_calibration(B, out / "figures" / "fig5_calibration_95ci.png")


def fig_calibration(B: pd.DataFrame, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from make_figures_s4 import STYLE
    from make_report import INK, MUTED, SURFACE
    models = [m for m in STYLE if m in set(B.model)]
    eps = ["OS", "RFS", "LR", "DM"]
    fig, axes = plt.subplots(len(eps), len(COHORTS), figsize=(9.5, 17))
    for i, ep in enumerate(eps):
        for j, co in enumerate(COHORTS):
            ax = axes[i, j]
            g = B[(B.cohort == co) & (B.endpoint == ep)]
            if g.empty:
                ax.axis("off")
                continue
            lo = max(0.0, min(g.predicted.min(), g.obs_lo.min()) - 0.03)
            ax.plot([lo, 1.0], [lo, 1.0], color=MUTED, lw=1, ls=(0, (3, 2)), zorder=1)
            for k, m in enumerate(models):
                h = g[g.model == m].sort_values("predicted")
                if h.empty:
                    continue
                col, mk, ls, _ = STYLE[m]
                dx = (k - (len(models) - 1) / 2) * 0.004     # small dodge so whiskers don't overlap
                ax.vlines(h.predicted + dx, h.obs_lo, h.obs_hi, color=col, lw=1.4, alpha=0.8, zorder=2)
                ax.plot(h.predicted + dx, h.observed, color=col, ls=ls, lw=1.4, marker=mk, ms=6,
                        mec=SURFACE, mew=1.2, zorder=3,
                        label=m if (i == 0 and j == 0) else None)
            ax.set_xlim(lo, 1.0); ax.set_ylim(lo, 1.0); ax.set_aspect("equal")
            ax.set_title(f"{ep} — {co}", fontsize=10, color=INK, pad=6, loc="left")
            if j == 0:
                ax.set_ylabel("Observed 2-year survival (KM, 95% CI)")
            if i == len(eps) - 1:
                ax.set_xlabel("Predicted 2-year survival")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper left", ncol=len(l), fontsize=8.5, bbox_to_anchor=(0.02, 0.972))
    fig.suptitle("Calibration at 2 years on the held-out test sets (dashed = perfect)",
                 x=0.02, ha="left", fontsize=12, color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"      {path}")


if __name__ == "__main__":
    main()
