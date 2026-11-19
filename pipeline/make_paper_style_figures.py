"""Figures 2, 3, 4 and Tables S4, S5 of the revision, drawn in the submitted paper's style.

Data: the merged report of the main run (OS/RFS on the submitted data, LR/DM on the corrected events).
Style: copied from the scripts that drew the submitted figures
  Fig 2/3: working_space/plot_fewshot_single_cohort_combine_both.py   (colours, markers, fonts, panel titles, grid)
  Fig 4  : working_space/infer_subgroups_single_cohort_three_models_single_Cohort_FINALLLL_.py
Changes from the submitted figures:
  - error bars (Fig 2) and bands (Fig 3) are 95% bootstrap CIs, not the SD across seeds
  - two new models: Clinical only, Clinical + CT-FM-ST (6 clinical variables + the out-of-fold CT-FM-ST score)
  - SPECTRE only for OS (as in the paper); never in the subgroup figure (as in the paper)
  - endpoints labelled OS / LR / DM / RFS (the submitted Fig 4 said EFS); y-axis has head-room so labels don't clip
Usage:
    python3 make_paper_style_figures.py [--report <outputs.dir>/<run>_study/report] [--final <folder>]
Writes <report>/paper_style/ and copies the renamed files to --final (default: config.yaml outputs.final_dir).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch

from ctfm_config import CFG, outputs_dir, paper_run

HERE = Path(__file__).resolve().parent
DEFAULT_REPORT = paper_run(must_exist=False)[0] / "report"     # outputs.paper_run, or the newest run

COMBINED = "Clinical + CT-FM-ST"
ORDER = ["CT-FM-ST", "CT-FM-LP", "SegRes-E2E", "SPECTRE", "Clinical only", COMBINED]
STYLE = {                       # first four: the submitted figures' MODEL_STYLES
    "CT-FM-ST":      ("#003f5c", "o", "-"),
    "CT-FM-LP":      ("#ff7f0e", "s", "-"),
    "SegRes-E2E":    ("#d62728", "D", "-"),
    "SPECTRE":       ("#9467bd", "^", "--"),
    "Clinical only": ("#7f7f7f", "X", ":"),
    COMBINED:        ("#17becf", "v", "-."),
}
EPS = ["OS", "LR", "DM", "RFS"]           # the submitted row order
COHORTS = ["Radcure", "Maastro"]
DISPLAY = {"LR": "LRF"}       # figure labels: locoregional failure (data keep the code name LR)
lab = lambda ep: DISPLAY.get(ep, ep)
FS = {"title": 18, "axis_label": 18, "tick_label": 16, "legend": 13, "annotation": 15, "grid_label": 15}


def load(report: Path):
    s4 = pd.read_csv(report / "table_s4_cindex_long.csv", dtype={"train_size": str})
    s5 = pd.read_csv(report / "table_s5_subgroups_long.csv")
    for d in (s4, s5):
        d.replace({"model": {"Clinical + CT-FM": COMBINED}}, inplace=True)
    s4 = s4[(s4.model != "SPECTRE") | (s4.endpoint == "OS")].copy()
    s5 = s5[s5.model != "SPECTRE"].copy()
    return s4, s5


def grid_labels(fig, axes):
    """Bold cohort headers above the columns and endpoint labels left of the rows (as the submitted grids)."""
    for c, co in enumerate(COHORTS):
        b = axes[0][c].get_position()
        fig.text((b.x0 + b.x1) / 2, b.y1 + 0.022, co, ha="center", va="bottom", fontsize=FS["grid_label"], weight="bold")
    for r, ep in enumerate(EPS):
        b = axes[r][0].get_position()
        fig.text(0.008, (b.y0 + b.y1) / 2, lab(ep), ha="left", va="center", fontsize=FS["grid_label"], weight="bold")


def save(fig, out: Path, name: str):
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"+ {out / name}.png/.pdf")


def fig2(s4: pd.DataFrame, out: Path):
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(4, 2, figsize=(19, 26), squeeze=False)
    for r, ep in enumerate(EPS):
        for c, co in enumerate(COHORTS):
            ax = axes[r][c]
            g = s4[(s4.endpoint == ep) & (s4.cohort == co) & (s4.train_size == "full")].set_index("model")
            ms = [m for m in ORDER if m in g.index]
            v, lo, hi = (g.loc[ms, k].to_numpy() for k in ("value", "lo", "hi"))
            x = np.arange(len(ms))
            ax.bar(x, v, yerr=[v - lo, hi - v], color=[STYLE[m][0] for m in ms], capsize=5,
                   edgecolor="black", linewidth=0.5, error_kw={"elinewidth": 1.2})
            for xi, vi, hi_ in zip(x, v, hi):
                ax.text(xi, hi_ + 0.02, f"{vi:.3f}", ha="center", va="bottom", fontsize=FS["annotation"])
            ax.set_xticks(x)
            ax.set_xticklabels([m.replace(" + ", " +\n") for m in ms], rotation=15, ha="right",
                               fontsize=FS["tick_label"])
            ax.set_ylabel("C-index", fontsize=FS["axis_label"])
            ax.set_ylim(0, 1.25)
            ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
            ax.tick_params(axis="y", labelsize=FS["tick_label"])
            n = int(g.n_train_full.iloc[0])
            ax.set_title(f"{co} - {lab(ep)} (Fullshot, N = {n})", fontsize=FS["title"], pad=12)
            ax.legend(handles=[Patch(facecolor=STYLE[m][0], edgecolor="black", linewidth=0.5, label=m) for m in ms],
                      loc="upper center", ncol=3, frameon=True, fontsize=FS["legend"])
    fig.tight_layout(rect=(0.03, 0, 1, 0.985), h_pad=2.5)
    grid_labels(fig, axes)
    save(fig, out, "fig2_fullshot_paper_style")


def fig3(s4: pd.DataFrame, out: Path):
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(4, 2, figsize=(20, 24), squeeze=False)
    for r, ep in enumerate(EPS):
        for c, co in enumerate(COHORTS):
            ax = axes[r][c]
            g = s4[(s4.endpoint == ep) & (s4.cohort == co)].copy()
            g["n"] = np.where(g.train_size == "full", g.n_train_full, pd.to_numeric(g.train_size, errors="coerce"))
            for m in [m for m in ORDER if m in set(g.model)]:
                h = g[g.model == m].sort_values("n")
                col, mk, ls = STYLE[m]
                ax.plot(h.n, h.value, label=m, color=col, marker=mk, markersize=8, linestyle=ls, linewidth=2)
                ax.fill_between(h.n, h.lo, h.hi, alpha=0.12, color=col, linewidth=0)
            ax.set_xlabel("Training samples (N)", fontsize=FS["axis_label"])
            ax.set_ylabel("Test C-index (mean, 95% CI)", fontsize=FS["axis_label"])
            ax.set_ylim(0.2, 1.12)
            ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
            ax.tick_params(axis="both", labelsize=FS["tick_label"])
            ax.set_title(f"{co} - {lab(ep)} (Few-shot)", fontsize=FS["title"], pad=12)
            ax.legend(loc="upper center", ncol=3, frameon=True, fontsize=FS["legend"])
    fig.tight_layout(rect=(0.03, 0, 1, 0.985), h_pad=2.5)
    grid_labels(fig, axes)
    save(fig, out, "fig3_fewshot_paper_style")


def order_subgroups(sgs):
    return ["All"] + sorted(s for s in sgs if s != "All") if "All" in sgs else sorted(sgs)


def fig4(s5: pd.DataFrame, out: Path):
    matplotlib.rcdefaults()                 # the submitted Fig 4 used plain matplotlib, not seaborn whitegrid
    models = [m for m in ORDER if m in set(s5.model)]
    k = len(models)
    width = 0.8 / k
    fig, axes = plt.subplots(4, 2, figsize=(24, 18), squeeze=False)
    for r, ep in enumerate(EPS):
        for c, co in enumerate(COHORTS):
            ax = axes[r][c]
            g = s5[(s5.endpoint == ep) & (s5.cohort == co)]
            sgs = order_subgroups(list(g.subgroup.unique()))
            x = np.arange(len(sgs))
            for i, m in enumerate(models):
                h = g[g.model == m].set_index("subgroup").reindex(sgs)
                pos = x + (i - (k - 1) / 2) * width
                ok = h.cindex.notna().to_numpy()
                ax.bar(pos[ok], h.cindex[ok], width, label=m, color=STYLE[m][0], edgecolor="black", linewidth=0.5)
                for xi, v in zip(pos[ok], h.cindex[ok]):
                    ax.text(xi, v + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=6.5, rotation=90)
            n = g.drop_duplicates("subgroup").set_index("subgroup").n.reindex(sgs)
            ax.set_xticks(x)
            ax.set_xticklabels([f"{s}\n(N={int(v)})" for s, v in n.items()], rotation=35, ha="right", fontsize=9)
            ax.tick_params(axis="y", labelsize=9)
            ax.set_ylim(0, 1.18)
            ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
            ax.set_ylabel("C-index", fontsize=12)
            ax.xaxis.grid(False)
            ax.yaxis.grid(True, linestyle="-", alpha=0.3)
            ax.set_axisbelow(True)
            ax.set_title(f"{co.upper()} - {lab(ep)} Subgroup Comparison", fontsize=14)
            ax.legend(fontsize=8, loc="upper center", ncol=k, frameon=True)
    fig.tight_layout()
    save(fig, out, "fig4_subgroups_paper_style")


def table_s5_md(s5: pd.DataFrame) -> str:
    models = [m for m in ORDER if m in set(s5.model)]
    L = ["## Table S5 (revision): subgroup C-index (95% CI), full shot, held-out test sets", "",
         "OS and RFS: the submitted data; LR and DM: corrected (pure) LR/DM events. 95% CI: 2,000 bootstrap resamples "
         "of the subgroup's test patients (all 5 runs jointly for CT-FM-LP and SegRes-E2E). — = too few events.", "",
         "| Endpoint | Cohort | Subgroup | N (events) | " + " | ".join(models) + " |",
         "|---|---|---|---|" + "---|" * len(models)]
    for ep in EPS:
        for co in COHORTS:
            g = s5[(s5.endpoint == ep) & (s5.cohort == co)]
            for sg in order_subgroups(list(g.subgroup.unique())):
                h = g[g.subgroup == sg].set_index("model")
                cells = []
                for m in models:
                    if m not in h.index or pd.isna(h.loc[m, "cindex"]):
                        cells.append("—")
                    else:
                        x = h.loc[m]
                        cells.append(f"{x.cindex:.2f} ({x.lo:.2f}–{x.hi:.2f})" if np.isfinite(x.lo) else f"{x.cindex:.2f}")
                f = h.iloc[0]
                L.append(f"| {ep} | {co} | {sg} | {int(f.n)} ({int(f.events)}) | " + " | ".join(cells) + " |")
    return "\n".join(L) + "\n"


S6_EPS = ["OS", "LR", "RFS", "DM"]        # the submitted Table S6 row order
S6_MODELS = ["CT-FM-ST", "CT-FM-LP", "SegRes-E2E", "Clinical only", COMBINED]   # IBS / calibration
EXT_IBS_MODELS = {"CT-FM-ST", "Clinical only", COMBINED}                       # external IBS
# SegRes-E2E IBS / calibration for OS and RFS: the submitted study's own saved network, scored by
# score_paper_networks_ibs.py. Value = which input to read from its summary:
# "submitted" (the paper's calibration input) or "hu_wide" (-1024..2048); None = keep the re-trained networks.
SEGRES_PAPER_NET = {"RFS": "submitted", "OS": "submitted"}   # OS also as submitted
PAPER_NET_DIR = outputs_dir() / "paper_networks_ibs"
# config.yaml submitted.use_saved_segres_networks: those networks exist only with the submitted study's saved
# models (optional/score_paper_networks_ibs.py). auto = use them when PAPER_NET_DIR has them, else the
# re-trained networks of the run;  on = required;  off = never.
_USE_SAVED = CFG.get("submitted.use_saved_segres_networks", "auto")
_USE_SAVED = {True: "on", False: "off"}.get(_USE_SAVED, str(_USE_SAVED).lower())
if _USE_SAVED == "off" or (_USE_SAVED == "auto" and not (PAPER_NET_DIR / "summary.csv").exists()):
    SEGRES_PAPER_NET = {k: None for k in SEGRES_PAPER_NET}
elif not (PAPER_NET_DIR / "summary.csv").exists():
    raise SystemExit(f"submitted.use_saved_segres_networks = on, but {PAPER_NET_DIR}/summary.csv does not exist "
                     f"-- run optional/score_paper_networks_ibs.py first")
EXT_TAG = "Radcure_to_Maastro_refit_train_primary"
ARM = {"clinical_only": "Clinical only", "ctfm_st": "CT-FM-ST", "combined_A": COMBINED}
ci = lambda v, lo, hi, d=3: f"{v:.{d}f} ({lo:.{d}f}–{hi:.{d}f})" if np.isfinite(lo) else f"{v:.{d}f}"


def table_s6(report: Path, s4: pd.DataFrame):
    """Submitted Table S6 layout (C-index, IBS, calibration error), now with 95% CIs; full shot, test sets.
    OS/RFS from report_original, LR/DM from report_corrected (the same split as Table S4)."""
    root = report.parent
    parts = []
    for d, eps in [("original", ["OS", "RFS"]), ("corrected", ["LR", "DM"])]:
        f = root / f"report_{d}" / "table_s6_calibration_long.csv"
        x = pd.read_csv(f)
        parts.append(x[x.endpoint.isin(eps)])
    L = _with_paper_segres(pd.concat(parts, ignore_index=True).replace({"model": {"Clinical + CT-FM": COMBINED}}))
    full = s4[s4.train_size == "full"].set_index(["endpoint", "cohort", "model"])
    rows = []
    for ep in S6_EPS:
        for co in COHORTS:
            for m in S6_MODELS:
                r = L[(L.endpoint == ep) & (L.cohort == co) & (L.model == m)]
                if r.empty:
                    continue
                r = r.iloc[0]
                c = full.loc[(ep, co, m)]
                rows.append({"Endpoint": ep, "Cohort": co + ("*" if co == "Maastro" else ""), "Model": m,
                             "N test (events)": f"{int(r.n_test)} ({int(r.n_events)})",
                             "C-index (95% CI)": ci(c.value, c.lo, c.hi),
                             "IBS (95% CI)": ci(r.ibs, r.ibs_lo, r.ibs_hi),
                             "Calibration error (2-yr)": f"{r.calib_error:.3f}",
                             "_ibs": r.ibs, "_ibs_lo": r.ibs_lo, "_ibs_hi": r.ibs_hi})
    T = pd.DataFrame(rows)
    note = ("Full-shot models on the held-out test sets. C-index as in Table S4. IBS: integrated Brier score (IPCW), "
            "lower = better. Calibration error: mean absolute difference between predicted and Kaplan-Meier observed "
            "2-year survival over bins of predicted risk (quintiles; tertiles for Maastro), point estimate only. "
            "95% CI: 2,000 event-stratified bootstrap resamples of the test patients (models with several training "
            "runs: mean over runs). OS and RFS: the submitted data; LR and DM: corrected events, death censored. "
            "*Maastro test set n = 26 (13 OS/RFS events, 4 LR/DM events): interpret with caution.")
    # SegRes-E2E OS/RFS rows come from the submitted study's saved network (SEGRES_PAPER_NET); not marked in the
    # table or figure.
    return T, note


def _paper_net_tertiles(var: str, ep_code: str, cohort: str = "Maastro"):
    """Calibration groups for the submitted SegRes-E2E network on Maastro: 3 equal-size groups by predicted 2-year
    survival with the KM 95% CI -- the same grouping make_calibration.py uses for every other Maastro panel."""
    import make_calibration as MC
    from lifelines import KaplanMeierFitter
    z = np.load(PAPER_NET_DIR / f"{var}_{cohort}_{ep_code}_SegRes-E2E.npz")
    pred, t, e = z["surv730"], z["test_times"], z["test_events"]
    rows = []
    for i, b in enumerate(MC._bins(pred, 3)):
        ci_ = KaplanMeierFitter().fit(t[b], e[b]).confidence_interval_survival_function_
        at = ci_.index[ci_.index <= MC.HORIZON]
        lo, hi = (ci_.loc[at[-1]].values if len(at) else (1.0, 1.0))
        rows.append({"bin": i, "predicted": float(pred[b].mean()), "observed": MC.km_at(t[b], e[b]),
                     "obs_lo": float(lo), "obs_hi": float(hi), "n": len(b)})
    return rows, MC.calib_error(pred, t, e, 3)


# SegRes-E2E LR/DM survival probabilities from the submitted refit step (StandardScaler + ridge Cox,
# penalizer 1.0, on the trained network's pooled features), applied to the re-trained LR/DM networks of this run
# (refit_segres_lrf_dm.py) -- so all four endpoints are computed like the submitted OS/RFS bundles above.
SEGRES_REFIT = {"LR": "lr", "DM": "dr"}
REFIT_DIR = outputs_dir() / "segres_refit_lrf_dm"
REFIT_NAME = {"LR": "LRF", "DM": "DM"}


def _refit_maastro(ep_code: str):
    """Maastro (n = 26): 3 equal-size groups of the seed-averaged 2-year survival, as every other Maastro panel;
    calibration error = mean over the 5 seeds (make_calibration.py convention)."""
    import make_calibration as MC
    from lifelines import KaplanMeierFitter
    z = np.load(REFIT_DIR / f"refit_Maastro_{ep_code}_SegRes-E2E.npz")
    pm, t, e, P = z["surv730"], z["test_times"], z["test_events"], z["surv730_per_seed"]
    rows = []
    for i, b in enumerate(MC._bins(pm, 3)):
        ci_ = KaplanMeierFitter().fit(t[b], e[b]).confidence_interval_survival_function_
        at = ci_.index[ci_.index <= MC.HORIZON]
        lo, hi = (ci_.loc[at[-1]].values if len(at) else (1.0, 1.0))
        rows.append({"bin": i, "predicted": float(pm[b].mean()), "observed": MC.km_at(t[b], e[b]),
                     "obs_lo": float(lo), "obs_hi": float(hi), "n": len(b)})
    return rows, float(np.mean([MC.calib_error(p, t, e, 3) for p in P]))


def _with_refit_segres(L: pd.DataFrame, bins: bool = False) -> pd.DataFrame:
    """Swap the SegRes-E2E LR/DM rows (IBS + calibration, or calibration bins) for the refit ones."""
    src = pd.read_csv(REFIT_DIR / ("bins.csv" if bins else "summary.csv"))
    for ep, code in SEGRES_REFIT.items():
        keep = ~((L.model == "SegRes-E2E") & (L.endpoint == ep))
        m_rows, m_ce = _refit_maastro(code)
        if bins:
            r = src[(src.endpoint == REFIT_NAME[ep]) & (src.cohort == "Radcure")].drop(columns="variant")
            new = pd.concat([r.assign(endpoint=ep),
                             pd.DataFrame([{"cohort": "Maastro", "endpoint": ep, "model": "SegRes-E2E", **x}
                                           for x in m_rows])], ignore_index=True)
        else:
            new = src[src.endpoint == REFIT_NAME[ep]][["cohort", "model", "ibs", "ibs_lo", "ibs_hi",
                                                       "calib_error"]].assign(endpoint=ep)
            new.loc[new.cohort == "Maastro", "calib_error"] = m_ce
            old = L[~keep].set_index("cohort")
            new = new.assign(n_test=new.cohort.map(old.n_test), n_events=new.cohort.map(old.n_events))
        L = pd.concat([L[keep], new[[c for c in L.columns if c in new.columns]]], ignore_index=True)
    return L


def _with_paper_segres(L: pd.DataFrame, bins: bool = False) -> pd.DataFrame:
    """Swap the SegRes-E2E rows (IBS + calibration, or calibration bins) for the endpoints in SEGRES_PAPER_NET
    with the submitted study's saved networks (score_paper_networks_ibs.py); LR/DM from the refit (above)."""
    if not any(SEGRES_PAPER_NET.values()):
        return _with_refit_segres(L, bins)
    src = pd.read_csv(PAPER_NET_DIR / ("bins.csv" if bins else "summary.csv"))
    for ep, var in SEGRES_PAPER_NET.items():
        if not var:
            continue
        keep = ~((L.model == "SegRes-E2E") & (L.endpoint == ep))
        new = src[(src.model == "SegRes-E2E") & (src.endpoint == ep) & (src.variant == var)].drop(columns="variant")
        # Maastro (n = 26): 3 equal-size groups, like every other Maastro panel
        m_bins, m_ce = _paper_net_tertiles(var, CODE[ep])
        if bins:
            new = pd.concat([new[new.cohort != "Maastro"],
                             pd.DataFrame([{"cohort": "Maastro", "endpoint": ep, "model": "SegRes-E2E", **r}
                                           for r in m_bins])], ignore_index=True)
        else:
            new.loc[new.cohort == "Maastro", "calib_error"] = m_ce
        if not bins:
            old = L[~keep].set_index("cohort")
            new = new.assign(n_test=new.cohort.map(old.n_test), n_events=new.cohort.map(old.n_events))
        L = pd.concat([L[keep], new[[c for c in L.columns if c in new.columns]]], ignore_index=True)
    return _with_refit_segres(L, bins)


def external(report: Path):
    ext = report / "external"
    R = pd.read_csv(ext / f"external_results_{EXT_TAG}.csv")
    C = pd.read_csv(ext / f"external_comparisons_{EXT_TAG}.csv")
    R = R[R.arm.isin(ARM)].assign(model=lambda x: x.arm.map(ARM))
    C = C[C.metric == "cindex"].assign(a=lambda x: x.model_a.map(ARM), b=lambda x: x.model_b.map(ARM))
    rows = []
    for ep in EPS:
        for m in [m for m in ORDER if m in set(R.model)]:
            r = R[(R.endpoint == ep) & (R.model == m)].iloc[0]
            row = {"Endpoint": ep, "Model": m, "N (events)": f"{int(r.n_target)} ({int(r.n_events)})",
                   "C-index (95% CI)": ci(r.cindex, r.cindex_lo, r.cindex_hi),
                   "IBS (95% CI)": ci(r.ibs, r.ibs_lo, r.ibs_hi) if m in EXT_IBS_MODELS else "",
                   "ΔC-index vs Clinical only (95% CI), p": "—"}
            k = C[(C.endpoint == ep) & (C.a == m) & (C.b == "Clinical only")]
            if len(k):
                k = k.iloc[0]
                row["ΔC-index vs Clinical only (95% CI), p"] = \
                    f"{k.delta:+.3f} ({k.lo:+.3f} to {k.hi:+.3f}), p = {k.p:.3f}"
            rows.append(row)
    note = ("External validation: models trained on all 940 Radcure training patients, tested on all 173 Maastro "
            "patients. Clinical + CT-FM-ST = the 6 clinical variables + the out-of-fold CT-FM-ST risk score. "
            "95% CI: 2,000 event-stratified bootstrap resamples of the Maastro patients; Δ and p from the paired "
            "bootstrap. IBS: integrated Brier score, lower = better. "
            "OS and RFS: the submitted data; LR and DM: corrected events, death censored.")
    return R, C, pd.DataFrame(rows), note


def md_table(T: pd.DataFrame, title: str, note: str) -> str:
    cols = [c for c in T.columns if not c.startswith("_")]
    L = [f"## {title}", "", note, "", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    L += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for _, r in T.iterrows()]
    return "\n".join(L) + "\n"


def fig_external_bars(R: pd.DataFrame, C: pd.DataFrame, out: Path):
    """Figure 2 style, one panel per endpoint: Radcure-trained models tested on all 173 Maastro patients."""
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(19, 14), squeeze=False)
    for ax, ep in zip(axes.ravel(), EPS):
        g = R[R.endpoint == ep].set_index("model")
        ms = [m for m in ORDER if m in g.index]
        v, lo, hi = (g.loc[ms, k].to_numpy() for k in ("cindex", "cindex_lo", "cindex_hi"))
        x = np.arange(len(ms))
        ax.bar(x, v, yerr=[v - lo, hi - v], color=[STYLE[m][0] for m in ms], capsize=5, width=0.6,
               edgecolor="black", linewidth=0.5, error_kw={"elinewidth": 1.2})
        for xi, vi, hi_ in zip(x, v, hi):
            ax.text(xi, hi_ + 0.02, f"{vi:.3f}", ha="center", va="bottom", fontsize=FS["annotation"])
        ax.set_xticks(x)
        ax.set_xticklabels(ms, rotation=15, ha="right", fontsize=FS["tick_label"])
        ax.set_ylabel("C-index", fontsize=FS["axis_label"])
        ax.set_ylim(0, 1.25)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.tick_params(axis="y", labelsize=FS["tick_label"])
        ev = int(g.n_events.iloc[0])
        ax.set_title(f"Radcure → Maastro - {lab(ep)} (External, N = {int(g.n_target.iloc[0])}, {ev} events)",
                     fontsize=FS["title"], pad=12)
        ax.legend(handles=[Patch(facecolor=STYLE[m][0], edgecolor="black", linewidth=0.5, label=m) for m in ms],
                  loc="upper center", ncol=3, frameon=True, fontsize=FS["legend"])
    fig.tight_layout(h_pad=3)
    save(fig, out, "figE_external_cindex_paper_style")


def fig_external_forest(R: pd.DataFrame, out: Path):
    """C-index and IBS with 95% CI per endpoint and model (forest layout), in the paper's colours."""
    sns.set_style("whitegrid")
    ms = [m for m in ORDER if m in set(R.model)]
    fig, axes = plt.subplots(1, 2, figsize=(16, 8.5), sharey=True)
    gap = 0.26
    for ax, (k, xl) in zip(axes, [("cindex", "C-index (higher = better)"),
                                  ("ibs", "Integrated Brier score (lower = better)")]):
        for r, ep in enumerate(EPS):
            for i, m in enumerate(ms):
                x = R[(R.endpoint == ep) & (R.model == m)].iloc[0]
                y = r + (i - (len(ms) - 1) / 2) * gap
                col, mk, _ = STYLE[m]
                ax.plot([x[f"{k}_lo"], x[f"{k}_hi"]], [y, y], color=col, linewidth=2.5)
                ax.plot(x[k], y, marker=mk, color=col, markersize=10, markeredgecolor="white",
                        linestyle="none", label=m if r == 0 else None)
                ax.text(x[f"{k}_hi"] + (0.008 if k == "cindex" else 0.003), y, f"{x[k]:.3f}",
                        va="center", fontsize=12)
        ax.set_xlabel(xl, fontsize=15)
        ax.tick_params(labelsize=13)
        ax.yaxis.grid(False)
    axes[0].set_yticks(range(len(EPS)))
    axes[0].set_yticklabels([lab(e) for e in EPS], fontsize=15, weight="bold")
    axes[0].invert_yaxis()
    axes[0].set_xlim(right=axes[0].get_xlim()[1] + 0.04)
    axes[1].set_xlim(right=axes[1].get_xlim()[1] + 0.012)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(l), fontsize=13, frameon=True, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("External validation: trained on Radcure (N = 940), tested on all 173 Maastro patients "
                 "(95% bootstrap CI)", fontsize=16)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    save(fig, out, "figE_external_forest_paper_style")


def _calibration_long(report: Path, name: str) -> pd.DataFrame:
    """OS/RFS rows from report_original, LR/DM rows from report_corrected (make_calibration outputs)."""
    parts = []
    for d, eps in [("original", ["OS", "RFS"]), ("corrected", ["LR", "DM"])]:
        x = pd.read_csv(report.parent / f"report_{d}" / name)
        parts.append(x[x.endpoint.isin(eps)])
    return pd.concat(parts, ignore_index=True).replace({"model": {"Clinical + CT-FM": COMBINED}})


def fig5(report: Path, cohort: str, out: Path, name: str):
    """Submitted Figure 5a/5b layout (rows = endpoints, columns = models; panel title = CalibError | IBS),
    now with the two clinical models and the Kaplan-Meier 95% CI of every observed point."""
    matplotlib.rcdefaults()
    B = _with_paper_segres(_calibration_long(report, "fig5_calibration_bins.csv"), bins=True)
    L = _with_paper_segres(_calibration_long(report, "table_s6_calibration_long.csv"))
    B, L = B[B.cohort == cohort], L[L.cohort == cohort]
    paper_eps = [k for k, v in SEGRES_PAPER_NET.items() if v]
    fig, axes = plt.subplots(len(EPS), len(S6_MODELS), figsize=(4.4 * len(S6_MODELS), 4.9 * len(EPS)),
                             squeeze=False)
    for r, ep in enumerate(EPS):
        for c, m in enumerate(S6_MODELS):
            ax = axes[r][c]
            b = B[(B.endpoint == ep) & (B.model == m)].sort_values("predicted")
            s = L[(L.endpoint == ep) & (L.model == m)]
            ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=0.8)
            if len(b):
                col, mk, _ = STYLE[m]
                ax.errorbar(b.predicted, b.observed, yerr=[b.observed - b.obs_lo, b.obs_hi - b.observed],
                            color=col, marker="o", markersize=7, linewidth=1.6, capsize=3, elinewidth=0.9)
            if len(s):
                s = s.iloc[0]
                ax.set_title(f"{m}\nCalibError: {s.calib_error:.3f}  |  IBS: {s.ibs:.3f}",
                             fontsize=11, weight="bold")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.grid(alpha=0.3)
            ax.set_xlabel("Predicted Survival Probability", fontsize=11)
            ax.set_ylabel("Observed Survival (KM)", fontsize=11)
    n = int(L.n_test.iloc[0])
    bins = "quintiles" if n >= 100 else "tertiles"
    fig.suptitle(f"{cohort} calibration comparison (N_test={n})  |  Calibration at t=730 days",
                 fontsize=18, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.965), h_pad=4.5)
    for r, ep in enumerate(EPS):
        b = axes[r][0].get_position()
        fig.text(0.5, b.y1 + 0.034, lab(ep), ha="center", va="bottom", fontsize=15, weight="bold")
    # no footnote in the image: the explanation is in the figure caption
    save(fig, out, name)


KM_COL = {0: "#9ecae1", 1: "#4a90c2", 2: "#003f5c"}        # low -> high risk, ramp of the CT-FM-ST blue
KM_NAME = {0: "Low", 1: "Medium", 2: "High"}
KM_YLAB = {"OS": "Overall survival", "RFS": "Recurrence-free survival",
           "LR": "Freedom from locoregional failure", "DM": "Freedom from distant metastasis"}
CODE = {"OS": "os", "RFS": "efs", "LR": "lr", "DM": "dr"}


KM_MODELS = ["CT-FM-ST", "Clinical only", COMBINED]


def _km_grid(panels: dict, cohort_label: str, suptitle: str, out: Path, name: str):
    """4 x 3 grid of Kaplan-Meier curves by predicted-risk tertile. panels[(ep, model)] = (years, events, tertile
    0/1/2, log-rank p). Shared by the internal (test set) and external figures."""
    from lifelines import KaplanMeierFitter
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(len(EPS), len(KM_MODELS), figsize=(6 * len(KM_MODELS), 4.6 * len(EPS)),
                             sharex=True, sharey=True, squeeze=False)
    kmf = KaplanMeierFitter()
    for r, ep in enumerate(EPS):
        for c, m in enumerate(KM_MODELS):
            ax = axes[r][c]
            t, e, grp, p = panels[(ep, m)]
            for g in (0, 1, 2):
                k = grp == g
                kmf.fit(t[k], e[k])
                sf = kmf.survival_function_
                x, s = sf.index.values, sf.iloc[:, 0].values
                keep = x <= 5
                ax.step(np.r_[x[keep], 5], np.r_[s[keep], s[keep][-1]], where="post", color=KM_COL[g], lw=2.5,
                        label=f"{KM_NAME[g]} predicted risk" if (r == 0 and c == 0) else None)
            ax.text(0.04, 0.06, f"log-rank p = {p:.2g}" if p >= 1e-4 else "log-rank p < 0.0001",
                    transform=ax.transAxes, fontsize=14, weight="bold" if p < 0.05 else "normal")
            ax.set_title(f"{cohort_label} - {lab(ep)} | {m}", fontsize=15, pad=10)
            ax.set_ylim(0, 1.02)
            ax.set_xlim(0, 5)
            ax.tick_params(labelsize=13)
            if c == 0:
                ax.set_ylabel(KM_YLAB[ep], fontsize=14)
            if r == len(EPS) - 1:
                ax.set_xlabel("Years", fontsize=14)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=14, frameon=True, bbox_to_anchor=(0.5, 0.012))
    fig.suptitle(suptitle, fontsize=17)
    fig.tight_layout(rect=(0, 0.035, 1, 0.97))
    save(fig, out, name)


def fig_external_km(report: Path, out: Path):
    """External: the 173 Maastro patients by predicted-risk tertile (tertiles and log-rank p from run_external)."""
    root = report.parent
    pred_dir = {d: sorted(root.glob(f"results_external_{d}_*"))[-1] / "predictions" for d in ("original", "corrected")}
    km = pd.read_csv(report / "external" / f"external_km_tertiles_{EXT_TAG}.csv")
    inv = {v: k for k, v in ARM.items()}
    panels = {}
    for ep in EPS:
        d = "original" if ep in ("OS", "RFS") else "corrected"
        for m in KM_MODELS:
            z = np.load(pred_dir[d] / f"Radcure_to_Maastro_{CODE[ep]}_{inv[m]}_refit_train_primary.npz",
                        allow_pickle=True)
            p = km[(km.endpoint == ep) & (km.arm == inv[m])].logrank_p.iloc[0]
            panels[(ep, m)] = (z["times"] / 365.25, z["events"], z["tertile"], p)
    _km_grid(panels, "Maastro", "External validation (trained on Radcure, tested on all 173 Maastro patients): "
             "Kaplan-Meier curves by predicted-risk tertile", out, "figE_external_km_paper_style")


def _tertiles(risk: np.ndarray) -> np.ndarray:
    """0 = lowest third of predicted risk, 2 = highest (equal-sized groups, ties broken by order)."""
    g = np.empty(len(risk), dtype=int)
    g[np.argsort(risk, kind="stable")] = np.arange(len(risk)) * 3 // len(risk)
    return g


def _run_dirs(report: Path, d: str):
    run = report.parent.name.replace("_study", "")
    return report.parent / f"results_{d}_n8_{run}", report.parent / f"results_baselines_{d}_n8_{run}"


def fig_internal_km(report: Path, cohort: str, out: Path, name: str):
    """Internal test set (full-shot models) by predicted-risk tertile -- the plot Reviewer 2 asked for."""
    from lifelines.statistics import multivariate_logrank_test
    inv = {v: k for k, v in ARM.items()}
    panels, n = {}, None
    for ep in EPS:
        A, _ = _run_dirs(report, "original" if ep in ("OS", "RFS") else "corrected")
        for m in KM_MODELS:
            z = np.load(A / inv[m] / "predictions" / f"{cohort}_{CODE[ep]}_primary_fullshot.npz", allow_pickle=True)
            t, e, grp = z["times"], z["events"], _tertiles(z["risk"])
            p = multivariate_logrank_test(t, grp, e).p_value
            panels[(ep, m)] = (t / 365.25, e, grp, p)
            n = len(t)
    _km_grid(panels, cohort, f"{cohort} held-out test set (n = {n}): Kaplan-Meier curves by predicted-risk tertile",
             out, name)


KM_COHORT_COL = {"Radcure": "#3d5a80", "Maastro": "#e07a5f"}


def fig_km_cohort(out: Path):
    """Whole-cohort Kaplan-Meier curves per endpoint (all patients), with numbers at risk -- Reviewer 2's request
    and the check on the event counts. Descriptive, so it uses the corrected data for every endpoint:
    the submitted Radcure RFS dates put the RFS event after the LR / DM failure date for
    74 / 139 patients. OS and all Maastro endpoints are identical in both versions."""
    import cohort as CH
    from lifelines import KaplanMeierFitter
    from lifelines.plotting import add_at_risk_counts
    sns.set_style("whitegrid")
    saved = {co: CH.COHORTS[co]["csv"] for co in COHORTS}
    data, counts = {}, []
    try:
        for v, eps in [("corrected", EPS)]:
            for co in COHORTS:
                CH.COHORTS[co]["csv"] = CH._CSV[v][co]
                for ep in eps:
                    df = CH.build_cohort(co, CODE[ep])
                    data[(ep, co)] = (df.time.values / 365.25, df.event.values.astype(int))
                    counts.append({"endpoint": ep, "cohort": co, "n": len(df), "events": int(df.event.sum()),
                                   "data": v})
    finally:
        for co, f in saved.items():
            CH.COHORTS[co]["csv"] = f
    fig, axes = plt.subplots(2, 2, figsize=(17, 14), squeeze=False)
    for ax, ep in zip(axes.ravel(), EPS):
        fits, full = [], {}
        for co in COHORTS:
            t, e = data[(ep, co)]
            kmf = KaplanMeierFitter(label=co).fit(t, e)          # short label: also the at-risk row name
            kmf.plot_survival_function(ax=ax, color=KM_COHORT_COL[co], lw=2.5, ci_alpha=0.15)
            fits.append(kmf)
            full[co] = f"{co} (n = {len(t)}, {e.sum()} events)"
        ax.set_xlim(0, 5)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("Years", fontsize=14)
        ax.set_ylabel(KM_YLAB[ep], fontsize=14)
        ax.set_title(f"{lab(ep)}", fontsize=16, weight="bold", pad=10)
        ax.tick_params(labelsize=13)
        h, l = ax.get_legend_handles_labels()
        ax.legend(h, [full.get(x, x) for x in l], loc="lower left", fontsize=13, frameon=True)
        add_at_risk_counts(*fits, ax=ax, xticks=[0, 1, 2, 3, 4, 5], rows_to_show=["At risk"], fontsize=12)
    fig.suptitle("Kaplan-Meier curves, whole cohorts (shaded: 95% CI)", fontsize=18)
    fig.tight_layout(rect=(0, 0, 1, 0.97), h_pad=3)
    save(fig, out, "figS_km_cohort_paper_style")
    return pd.DataFrame(counts)


def rfs_sensitivity(report: Path):
    """RFS with the submitted vs the corrected Radcure event dates (186 dates corrected; events unchanged) --
    CT-FM-ST, Clinical only, Clinical + CT-FM-ST. Internal: the main run vs sensitivity_rfs_corrected_dates/
    (same pipeline, corrected data, full shot); external: the two external runs. Returns (table, note) or None."""
    root = report.parent
    run = root.name.replace("_study", "")
    sens = root / "sensitivity_rfs_corrected_dates" / f"results_corrected_n8_{run}" / "_cells"
    if not sens.exists():
        return None
    main = root / f"results_original_n8_{run}" / "_cells"
    ext = {d: sorted(root.glob(f"results_external_{d}_*"))[-1] for d in ("original", "corrected")}
    pfmt = lambda p: "< 0.001" if p < 0.001 else f"{p:.3f}"
    rows = []

    def add(setting, co, n, ev, label, a, b, kind):
        if kind == "c":
            cell = lambda r: ci(r["cindex"], r["cindex_lo"], r["cindex_hi"])
            diff = f"{b['cindex'] - a['cindex']:+.3f}"
            ibs = lambda r: ci(r["ibs"], r["ibs_lo"], r["ibs_hi"])
            rows.append({"Setting": setting, "Cohort": co, "N (events)": f"{n} ({ev})", "Model / comparison": label,
                         "C-index, submitted dates": cell(a), "C-index, corrected dates": cell(b), "Δ C-index": diff,
                         "IBS, submitted dates": ibs(a), "IBS, corrected dates": ibs(b)})
        else:
            fmt = lambda r: f"{r['delta']:+.3f} ({r['lo']:+.3f} to {r['hi']:+.3f}), p = {pfmt(r['p'])}"
            rows.append({"Setting": setting, "Cohort": co, "N (events)": f"{n} ({ev})", "Model / comparison": label,
                         "C-index, submitted dates": fmt(a), "C-index, corrected dates": fmt(b), "Δ C-index": "",
                         "IBS, submitted dates": "", "IBS, corrected dates": ""})

    for co in COHORTS:
        R = {k: pd.read_csv(p / f"rows_full_{co}_efs_primary.csv").set_index("arm") for k, p in
             (("a", main), ("b", sens))}
        Cp = {k: pd.read_csv(p / f"comps_full_{co}_efs_primary.csv") for k, p in (("a", main), ("b", sens))}
        n, ev = int(R["a"].n_test.iloc[0]), int(R["a"].n_events.iloc[0])
        for arm in ("ctfm_st", "clinical_only", "combined_A"):
            add("Internal", co, n, ev, ARM[arm], R["a"].loc[arm], R["b"].loc[arm], "c")
        pick = lambda C: C[(C.metric == "cindex") & (C.model_a == "combined_A") & (C.model_b == "clinical_only")].iloc[0]
        add("Internal", co, n, ev, f"{COMBINED} − Clinical only", pick(Cp["a"]), pick(Cp["b"]), "d")
    E = {d: pd.read_csv(p / f"external_results_{EXT_TAG}.csv") for d, p in ext.items()}
    EC = {d: pd.read_csv(p / f"external_comparisons_{EXT_TAG}.csv") for d, p in ext.items()}
    for arm in ("ctfm_st", "clinical_only", "combined_A"):
        a, b = (E[d][(E[d].endpoint == "RFS") & (E[d].arm == arm)].iloc[0] for d in ("original", "corrected"))
        add("External", "Radcure → Maastro", int(a.n_target), int(a.n_events), ARM[arm], a, b, "c")
    a, b = (EC[d][(EC[d].endpoint == "RFS") & (EC[d].metric == "cindex") & (EC[d].model_a == "combined_A") &
                  (EC[d].model_b == "clinical_only")].iloc[0] for d in ("original", "corrected"))
    add("External", "Radcure → Maastro", int(a.n_target), int(a.n_events), f"{COMBINED} − Clinical only", a, b, "d")
    note = ("Sensitivity analysis: recurrence-free survival with the submitted vs the corrected Radcure event dates "
            "(186 RFS event dates corrected so that RFS is never later than the first locoregional or distant "
            "failure; event status unchanged; Maastro dates unchanged). Same models and pipeline, full shot; "
            "external models are trained on Radcure, so the correction also affects them. 95% CI: 2,000 "
            "event-stratified bootstrap resamples; Δ rows: paired bootstrap.")
    return pd.DataFrame(rows), note


PAIRS = [("CT-FM-LP", "CT-FM-ST"), ("SegRes-E2E", "CT-FM-ST"), ("SPECTRE", "CT-FM-ST"),
         ("CT-FM-ST", "Clinical only"), (COMBINED, "Clinical only"), (COMBINED, "CT-FM-ST")]


def _test_predictions(report: Path, d: str, co: str, ep: str) -> tuple:
    """Every model's full-shot test-set risk scores for one cell: (times, events, {model: (seeds, n) array})."""
    A, Bd = _run_dirs(report, d)
    src = {"CT-FM-ST": A / "ctfm_st", "Clinical only": A / "clinical_only", COMBINED: A / "combined_A",
           "CT-FM-LP": Bd / "ctfm_lp", "SegRes-E2E": Bd / "segres_e2e"}
    if ep == "OS":
        src["SPECTRE"] = Bd / "spectre"
    t0 = e0 = None
    P = {}
    for m, p in src.items():
        z = np.load(p / "predictions" / f"{co}_{CODE[ep]}_primary_fullshot.npz", allow_pickle=True)
        k = int(z["n_seeds"]) if "n_seeds" in z.files else 1
        n = len(z["times"]) // k
        t, e = z["times"][:n], z["events"][:n]
        if t0 is None:
            t0, e0 = t, e
        assert np.array_equal(t, t0) and np.array_equal(e, e0), f"patient order differs: {m} {co} {ep}"
        P[m] = z["risk"].reshape(k, n)
    return t0, e0, P


def paired_comparisons(report: Path, out: Path, B: int = 2000, recompute: bool = False) -> pd.DataFrame:
    """Paired bootstrap of C-index differences between models, internal test sets (full shot) + external.
    Same procedure as bootstrap_ci.paired_bootstrap (event-stratified, seed 42, B = 2000, two-sided p);
    models with 5 runs enter as the mean C-index over runs in every resample, as in Table S4."""
    cache = out / "paired_comparisons_long.csv"
    if cache.exists() and not recompute:
        return pd.read_csv(cache)
    from lifelines.utils import concordance_index
    from bootstrap_ci import _resample_idx
    c_of = lambda t, e, R, i: float(np.mean([concordance_index(t[i], -r[i], e[i]) for r in R]))
    rows = []
    for ep in EPS:
        d = "original" if ep in ("OS", "RFS") else "corrected"
        A, _ = _run_dirs(report, d)
        for co in COHORTS:
            t, e, P = _test_predictions(report, d, co, ep)
            pairs = [(a, b) for a, b in PAIRS if a in P and b in P]
            full = np.arange(len(t))
            point = {m: c_of(t, e, R, full) for m, R in P.items()}
            rng = np.random.default_rng(42)
            deltas = {pr: [] for pr in pairs}
            for _ in range(B):
                idx = _resample_idx(rng, e, True)
                if e[idx].sum() < 2:
                    continue
                cs = {}
                for m, R in P.items():
                    try:
                        cs[m] = c_of(t, e, R, idx)
                    except (ZeroDivisionError, ValueError):
                        cs[m] = np.nan
                for a, b in pairs:
                    if np.isfinite(cs[a]) and np.isfinite(cs[b]):
                        deltas[(a, b)].append(cs[a] - cs[b])
            comps = pd.read_csv(A / "_cells" / f"comps_full_{co}_{CODE[ep]}_primary.csv")
            lrt = comps[comps.metric == "LRT"]
            for a, b in pairs:
                dd = np.asarray(deltas[(a, b)])
                lo, hi = np.percentile(dd, [2.5, 97.5])
                p = min(max(2 * min((dd <= 0).mean(), (dd >= 0).mean()), 1 / (len(dd) + 1)), 1.0)
                rows.append({"setting": "Internal", "endpoint": ep, "cohort": co, "n_test": len(t),
                             "n_events": int(e.sum()), "model_a": a, "model_b": b, "delta": point[a] - point[b],
                             "lo": lo, "hi": hi, "p": p,
                             "lrt_chi2": float(lrt.delta.iloc[0]) if (a == COMBINED and b == "Clinical only" and len(lrt)) else np.nan,
                             "lrt_p": float(lrt.p.iloc[0]) if (a == COMBINED and b == "Clinical only" and len(lrt)) else np.nan})
                if a == COMBINED and b == "Clinical only":          # must equal the run's own paired bootstrap
                    k = comps[(comps.metric == "cindex") & (comps.model_a == "combined_A") &
                              (comps.model_b == "clinical_only")].iloc[0]
                    assert abs(k.delta - rows[-1]["delta"]) < 1e-9 and abs(k.lo - lo) < 1e-9 and abs(k.p - p) < 1e-9, \
                        f"paired bootstrap differs from the run's: {co} {ep}"
            print(f"   paired comparisons: {co} {ep} done", flush=True)
    E = pd.read_csv(report / "external" / f"external_comparisons_{EXT_TAG}.csv")
    E = E[E.metric == "cindex"]
    for ep in EPS:
        for a, b in PAIRS:
            k = E[(E.endpoint == ep) & (E.model_a.map(ARM) == a) & (E.model_b.map(ARM) == b)]
            if len(k):
                k = k.iloc[0]
                rows.append({"setting": "External", "endpoint": ep, "cohort": "Radcure → Maastro",
                             "n_test": int(k.n_target), "n_events": int(k.n_events), "model_a": a, "model_b": b,
                             "delta": k.delta, "lo": k.lo, "hi": k.hi, "p": k.p, "lrt_chi2": np.nan, "lrt_p": np.nan})
    T = pd.DataFrame(rows)
    T.to_csv(cache, index=False)
    return T


def table_paired_md(T: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    pfmt = lambda p: "< 0.001" if p < 0.001 else f"{p:.3f}"
    W = pd.DataFrame([{"Setting": r.setting, "Endpoint": r.endpoint, "Cohort": r.cohort,
                       "N test (events)": f"{r.n_test} ({r.n_events})",
                       "Comparison": f"{r.model_a} − {r.model_b}",
                       "ΔC-index (95% CI)": f"{r.delta:+.3f} ({r.lo:+.3f} to {r.hi:+.3f})", "p": pfmt(r.p),
                       "LRT, adding CT-FM-ST to clinical (χ², p)":
                           f"{r.lrt_chi2:.1f}, {pfmt(r.lrt_p)}" if np.isfinite(r.lrt_chi2) else ""}
                      for r in T.itertuples()])
    note = ("Paired comparisons of the C-index, full-shot models. Δ = first model − second model (positive = the "
            "first discriminates better). 95% CI and two-sided p: paired bootstrap, 2,000 event-stratified "
            "resamples of the test patients, both models scored on the same resamples; CT-FM-LP, SegRes-E2E and "
            "SPECTRE enter as the mean over their 5 runs. LRT: likelihood-ratio test for adding the CT-FM-ST score "
            "to the clinical Cox model (training data). Internal: held-out test sets (Radcure n = 202, Maastro "
            "n = 26); External: trained on Radcure, tested on all 173 Maastro patients. OS and RFS: the submitted "
            "data; LR and DM: corrected events, death censored. SPECTRE: OS only.")
    return W, note

FINAL_DEFAULT = CFG.path("outputs.final_dir")
FINAL_NAMES = {   # paper_style/<name> -> final folder name
    "fig2_fullshot_paper_style": "Figure2_fullshot_cindex_95CI",
    "fig3_fewshot_paper_style": "Figure3_fewshot_curves_95CI",
    "fig4_subgroups_paper_style": "Figure4_subgroups_cindex",
    "fig5a_calibration_radcure_paper_style": "Figure5a_calibration_Radcure",
    "fig5b_calibration_maastro_paper_style": "Figure5b_calibration_Maastro",
    "figE_external_cindex_paper_style": "FigureExt_external_cindex_bars_95CI",
    "figE_external_forest_paper_style": "FigureExt_external_forest_cindex_ibs_95CI",
    "figE_external_km_paper_style": "FigureExt_external_km_tertiles",
    "figKM_internal_radcure_paper_style": "FigureKM_internal_tertiles_Radcure",
    "figKM_internal_maastro_paper_style": "FigureKM_internal_tertiles_Maastro",
    "figS_km_cohort_paper_style": "FigureKM_cohort_all_endpoints",
}
TABLE_NAMES = {
    "table_s4.md": "TableS4_internal_cindex_95CI.md", "table_s4_cindex_long.csv": "TableS4_internal_cindex_95CI.csv",
    "table_s5.md": "TableS5_subgroups_cindex_95CI.md", "table_s5_subgroups_long.csv": "TableS5_subgroups_cindex_95CI.csv",
    "table_s6.md": "TableS6_internal_ibs_calibration_95CI.md", "table_s6.csv": "TableS6_internal_ibs_calibration_95CI.csv",
    "table_external.md": "TableExt_external_validation_95CI.md", "table_external.csv": "TableExt_external_validation_95CI.csv",
    "table_paired.md": "TablePaired_model_comparisons_95CI.md", "table_paired.csv": "TablePaired_model_comparisons_95CI.csv",
    "table_rfs_sensitivity.md": "TableSens_RFS_corrected_dates.md", "table_rfs_sensitivity.csv": "TableSens_RFS_corrected_dates.csv",
}


def copy_final(out: Path, final: Path, run: str):
    import shutil
    (final / "figures").mkdir(parents=True, exist_ok=True)
    (final / "tables").mkdir(parents=True, exist_ok=True)
    for src, dst in FINAL_NAMES.items():
        for ext in ("png", "pdf"):
            shutil.copy2(out / f"{src}.{ext}", final / "figures" / f"{dst}.{ext}")
    for src, dst in TABLE_NAMES.items():
        shutil.copy2(out / src, final / "tables" / dst)
    (final / "README.md").write_text(
        "# Revision figures and tables (ready to use)\n\n"
        f"Made by `pipeline/make_paper_style_figures.py` from the run `{run}` "
        "(OS and RFS: the submitted data; LR and DM: corrected events). Every number has a 95% CI from 2,000 "
        "event-stratified bootstrap resamples of the test patients. Re-running the script overwrites this folder.\n\n"
        "| File | What | Replaces |\n|---|---|---|\n"
        "| figures/Figure2_fullshot_cindex_95CI | full-shot C-index, internal test sets, 6 models (SPECTRE: OS only) | Figure 2 |\n"
        "| figures/Figure3_fewshot_curves_95CI | few-shot learning curves with 95% CI bands | Figure 3 |\n"
        "| figures/Figure4_subgroups_cindex | subgroup C-index, full shot (CIs in Table S5) | Figure 4 |\n"
        "| figures/Figure5a_calibration_Radcure | 2-year calibration + IBS, 5 models, KM 95% CI per point | Figure 5a |\n"
        "| figures/Figure5b_calibration_Maastro | same, Maastro (tertiles, n = 26) | Figure 5b |\n"
        "| figures/FigureExt_external_cindex_bars_95CI | external validation, Figure 2 style (ΔC vs clinical: external table) | new |\n"
        "| figures/FigureExt_external_forest_cindex_ibs_95CI | external validation, C-index and IBS (all three models) | new |\n"
        "| figures/FigureExt_external_km_tertiles | external validation, KM by predicted-risk tertile, log-rank p | new |\n"
        "| figures/FigureKM_internal_tertiles_Radcure | internal test set, KM by predicted-risk tertile, log-rank p | new |\n"
        "| figures/FigureKM_internal_tertiles_Maastro | same, Maastro test set (n = 26) | new (supplement) |\n"
        "| figures/FigureKM_cohort_all_endpoints | whole-cohort KM per endpoint, numbers at risk | new (R2) |\n"
        "| tables/TableS4_internal_cindex_95CI | C-index at 10/24/50/100/full | Table S4 |\n"
        "| tables/TableS5_subgroups_cindex_95CI | subgroup C-index | Table S5 |\n"
        "| tables/TableS6_internal_ibs_calibration_95CI | C-index, IBS, calibration error (full shot, 5 models) | Table S6 |\n"
        "| tables/TableExt_external_validation_95CI | external C-index, IBS, ΔC vs clinical | new |\n"
        "| tables/TablePaired_model_comparisons_95CI | paired ΔC between models (internal + external), LRT | new (R3) |\n"
        "| tables/TableSens_RFS_corrected_dates | RFS with submitted vs corrected Radcure event dates | new (sensitivity) |\n"
        "\nModels use the submitted RFS dates (so the paper reproduces); the cohort Kaplan-Meier figure uses the "
        "corrected dates (descriptive); TableSens shows the difference.\n")
    print(f"-> final folder {final}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--report", default=str(DEFAULT_REPORT))
    ap.add_argument("--final", default=str(FINAL_DEFAULT), help="folder that receives the ready figures and tables")
    ap.add_argument("--recompute-paired", action="store_true", help="redo the paired bootstrap (cached otherwise)")
    a = ap.parse_args()
    report = Path(a.report)
    out = report / "paper_style"
    out.mkdir(exist_ok=True)
    s4, s5 = load(report)
    s4.to_csv(out / "table_s4_cindex_long.csv", index=False)
    s5.to_csv(out / "table_s5_subgroups_long.csv", index=False)
    import merged_report as PRR                    # reuse its Table S4 layout
    PRR.MODELS = ORDER
    (out / "table_s4.md").write_text(PRR.table_s4_md(s4).replace("(this run)", "(revision)"))
    (out / "table_s5.md").write_text(table_s5_md(s5))
    T6, note6 = table_s6(report, s4)
    T6.drop(columns=[c for c in T6 if c.startswith("_")]).to_csv(out / "table_s6.csv", index=False)
    (out / "table_s6.md").write_text(md_table(T6, "Table S6 (revision). Calibration and Brier score, full shot, "
                                              "with 95% CI", note6))
    R, C, TE, noteE = external(report)
    TE.to_csv(out / "table_external.csv", index=False)
    (out / "table_external.md").write_text(md_table(TE, "External validation (Radcure → Maastro), with 95% CI", noteE))
    TP, noteP = table_paired_md(paired_comparisons(report, out, recompute=a.recompute_paired))
    TP.to_csv(out / "table_paired.csv", index=False)
    (out / "table_paired.md").write_text(md_table(TP, "Paired model comparisons (C-index), with 95% CI", noteP))
    sens = rfs_sensitivity(report)
    if sens is not None:
        TR, noteR = sens
        TR.to_csv(out / "table_rfs_sensitivity.csv", index=False)
        (out / "table_rfs_sensitivity.md").write_text(md_table(TR, "RFS: submitted vs corrected Radcure event dates",
                                                               noteR))
    fig2(s4, out)
    fig3(s4, out)
    fig_external_bars(R, C, out)
    fig_external_forest(R, out)
    fig_external_km(report, out)
    fig_internal_km(report, "Radcure", out, "figKM_internal_radcure_paper_style")
    fig_internal_km(report, "Maastro", out, "figKM_internal_maastro_paper_style")
    fig_km_cohort(out).to_csv(out / "cohort_event_counts.csv", index=False)
    fig4(s5, out)                       # from here on: plain matplotlib (rcdefaults), as the submitted Fig 4/5
    fig5(report, "Radcure", out, "fig5a_calibration_radcure_paper_style")
    fig5(report, "Maastro", out, "fig5b_calibration_maastro_paper_style")
    print(f"-> {out}")
    copy_final(out, Path(a.final), report.parent.name.replace("_study", ""))


if __name__ == "__main__":
    main()
