"""Figures comparing the three models on the external cohort (train Radcure -> all Maastro).

Reads the outputs of run_external.py -- no model fitting, so it can be re-run any time.
run_external.py calls it automatically at the end of a run.

    python make_external_report.py                       # newest run
    python make_external_report.py --tag Radcure_to_Maastro_refit_train_primary

Figures (in results_external_<version>/figures/):
    ext_forest_<tag>.png   C-index and IBS with 95% CI, 3 models x 4 endpoints
    ext_delta_<tag>.png    C-index gain over clinical-only (paired bootstrap)
    ext_km_<tag>.png       Kaplan-Meier curves by predicted-risk tertile (R2-6)

Colour: models keep the categorical slots used everywhere else (blue / orange / aqua).
Risk tertiles are ORDERED, so they take a single-hue ordinal ramp -- blue steps
250 / 450 / 650 (#86b6ef / #2a78d6 / #104281), validated as an ordinal ramp: monotone
lightness, adjacent dL 0.19, light end 2.06:1, hue spread 3 deg. The light step is
below 3:1, so the relief rule applies: curves are direct-labelled, a legend is
present, and the summary .md written by run_external.py is the table view. A
lightness ramp also survives greyscale print.
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter

from cohort import DATA_VERSION, FEATURE_VERSION, IMAGING, latest_results_dir, results_dir
# shared style: importing make_report also applies its rcParams
from make_report import ARMS, COLOR, INK, INK2, LABEL, MARKER, MUTED, SURFACE

OUT = None            # run folder; newest results_external_<version>_<run id> by default
ENDPOINTS = ["OS", "RFS", "LR", "DM"]
CODE = {"OS": "os", "RFS": "efs", "LR": "lr", "DM": "dr"}
TERTILE_COLOR = {0: "#86b6ef", 1: "#2a78d6", 2: "#104281"}
TERTILE_NAME = {0: "Low", 1: "Medium", 2: "High"}
YLAB = {"OS": "Overall survival", "RFS": "Recurrence-free",
        "LR": "Free of LR failure", "DM": "Free of distant metastasis"}
HORIZON_YEARS = 5


def newest_tag() -> str:
    hits = sorted(glob.glob(str(OUT / "external_results_*.csv")), key=lambda p: Path(p).stat().st_mtime)
    if not hits:
        raise FileNotFoundError(f"no external_results_*.csv in {OUT} -- run run_external.py first")
    return Path(hits[-1]).stem.replace("external_results_", "")


def _blocks(ax, items, draw):
    """Lay out endpoint blocks of one row per model; returns y tick positions/labels."""
    ticks, labels, y = [], [], 0.0
    for ep in ENDPOINTS:
        start = y
        for arm in items:
            draw(ax, ep, arm, y)
            y += 1
        ticks.append((start + y - 1) / 2); labels.append(ep); y += 0.8
    ax.set_yticks(ticks); ax.set_yticklabels(labels, fontsize=9, color=INK)
    ax.invert_yaxis(); ax.grid(axis="y", visible=False)


def fig_forest(rows, path, src, tgt, title=None):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.6))
    for ax, metric, xlabel in [(axes[0], "cindex", "C-index (higher = better)"),
                               (axes[1], "ibs", "Integrated Brier Score (lower = better)")]:
        def draw(ax, ep, arm, y):
            r = rows[(rows.endpoint == ep) & (rows.arm == arm)]
            if r.empty:
                return
            r = r.iloc[0]
            p, lo, hi = r[metric], r[f"{metric}_lo"], r[f"{metric}_hi"]
            if not np.isfinite([p, lo, hi]).all():
                return
            ax.plot([lo, hi], [y, y], color=COLOR[arm], lw=2, solid_capstyle="round", zorder=2)
            ax.plot([p], [y], marker=MARKER[arm], ms=8, color=COLOR[arm], mec=SURFACE, mew=2,
                    zorder=3, label=LABEL[arm] if ep == ENDPOINTS[0] else None)
            ax.annotate(f"{p:.3f}", (hi, y), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=7.5, color=INK2)
        _blocks(ax, ARMS, draw)
        g = rows[np.isfinite(rows[f"{metric}_hi"])]
        if len(g):
            lo_x, hi_x = g[f"{metric}_lo"].min(), g[f"{metric}_hi"].max()
            pad = (hi_x - lo_x) * 0.2
            ax.set_xlim(min(lo_x - pad * 0.25, 0.48) if metric == "cindex" else lo_x - pad * 0.25,
                        hi_x + pad)
        if metric == "cindex":
            ax.axvline(0.5, color=MUTED, lw=0.8, ls=(0, (3, 3)), zorder=1)
        ax.set_xlabel(xlabel)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, -0.01))
    n = int(rows.n_target.iloc[0])
    fig.suptitle(title or f"External validation: trained on {src}, tested on all {n} {tgt} "
                 f"patients (95% bootstrap CI)", x=0.02, ha="left", fontsize=12, color=INK)
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])
    fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)


def fig_delta(comps, path, src, tgt, title=None):
    d = comps[(comps.metric == "cindex") & (comps.model_b == "clinical_only")]
    arms = [a for a in [IMAGING, "combined_A"] if a in set(d.model_a)]
    if d.empty or not arms:
        return False
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    def draw(ax, ep, arm, y):
        r = d[(d.endpoint == ep) & (d.model_a == arm)]
        if r.empty:
            return
        r = r.iloc[0]
        sig = np.isfinite(r.p) and r.p < 0.05
        ax.plot([r.lo, r.hi], [y, y], color=COLOR[arm], lw=2, solid_capstyle="round", zorder=2)
        ax.plot([r.delta], [y], marker=MARKER[arm], ms=8, color=COLOR[arm], mec=SURFACE, mew=2,
                zorder=3, label=f"{LABEL[arm]} − Clinical" if ep == ENDPOINTS[0] else None)
        ax.annotate(f"{r.delta:+.3f}" + ("*" if sig else ""), (r.hi, y), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=7.5,
                    color=INK if sig else INK2)
    _blocks(ax, arms, draw)
    ax.axvline(0, color=INK, lw=1.0, zorder=1)
    g = d[np.isfinite(d.hi)]
    pad = (g.hi.max() - g.lo.min()) * 0.2
    ax.set_xlim(g.lo.min() - pad * 0.25, g.hi.max() + pad)
    ax.set_xlabel("Δ C-index vs clinical-only on the external cohort")
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(title or f"Gain over the clinical model, {tgt} (paired bootstrap; * = p < 0.05)",
                 x=0.02, ha="left", fontsize=12, color=INK)
    fig.tight_layout(rect=[0, 0.08, 1, 0.93])
    fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)
    return True


def _declutter(ys, min_gap, lo=0.03, hi=1.0):
    """Nudge end-of-curve labels apart so they never overlap, keeping their order AND
    keeping them inside [lo, hi] -- when curves end near the top, the stack is pushed
    down instead of spilling above the axes."""
    order = np.argsort(ys)
    pos = np.array(ys, float)[order]
    for i in range(1, len(pos)):                     # upward pass: enforce spacing
        pos[i] = max(pos[i], pos[i - 1] + min_gap)
    if len(pos) and pos[-1] > hi:                    # too high: pin the top, push down
        pos[-1] = hi
        for i in range(len(pos) - 2, -1, -1):
            pos[i] = min(pos[i], pos[i + 1] - min_gap)
    pos = np.maximum(pos, lo)
    out = np.empty_like(pos)
    out[order] = pos
    return out


def fig_km(km, pred_path, path, tgt, title=None):
    fig, axes = plt.subplots(len(ENDPOINTS), len(ARMS), figsize=(11, 12.5),
                             sharex=True, sharey=True)
    kmf = KaplanMeierFitter()
    for i, ep in enumerate(ENDPOINTS):
        for j, arm in enumerate(ARMS):
            ax = axes[i, j]
            f = pred_path(CODE[ep], arm)
            if not Path(f).exists():
                ax.text(0.5, 0.5, "not run", ha="center", va="center",
                        transform=ax.transAxes, color=MUTED)
                continue
            z = np.load(f, allow_pickle=True)
            t_yr, e, grp = z["times"] / 365.25, z["events"], z["tertile"]
            ends = {}
            for g in (2, 1, 0):                       # draw high risk last = on top
                m = grp == g
                if m.sum() < 3:
                    continue
                kmf.fit(t_yr[m], e[m])
                sf = kmf.survival_function_
                x = sf.index.values; s = sf.iloc[:, 0].values
                keep = x <= HORIZON_YEARS
                x = np.r_[x[keep], HORIZON_YEARS]; s = np.r_[s[keep], s[keep][-1]]
                ax.step(x, s, where="post", color=TERTILE_COLOR[g], lw=2,
                        label=f"{TERTILE_NAME[g]} risk" if (i == 0 and j == 0) else None)
                ends[g] = s[-1]
            # direct labels at the right edge, nudged so they never collide
            gs = sorted(ends)
            for g, yv in zip(gs, _declutter([ends[g] for g in gs], 0.075)):
                ax.annotate(TERTILE_NAME[g], (HORIZON_YEARS, yv), xytext=(4, 0),
                            textcoords="offset points", va="center", fontsize=7.5, color=INK2,
                            annotation_clip=False)
            k = km[(km.endpoint == ep) & (km.arm == arm)]
            if len(k):
                p = k.iloc[0].logrank_p
                ax.text(0.04, 0.05, f"log-rank p = {p:.2g}", transform=ax.transAxes,
                        fontsize=8, color=INK if p < 0.05 else INK2)
            ax.set_ylim(0, 1.02); ax.set_xlim(0, HORIZON_YEARS)
            if i == 0:
                ax.set_title(LABEL[arm], fontsize=10, color=INK, pad=8)
            if j == 0:
                ax.set_ylabel(YLAB[ep], fontsize=9)
            if i == len(ENDPOINTS) - 1:
                ax.set_xlabel("years")
    h, l = axes[0, 0].get_legend_handles_labels()
    order = [f"{TERTILE_NAME[g]} risk" for g in (0, 1, 2)]          # Low -> Medium -> High
    pairs = sorted(zip(h, l), key=lambda hl: order.index(hl[1]) if hl[1] in order else 9)
    fig.legend([p[0] for p in pairs], [p[1] for p in pairs], loc="lower center", ncol=3,
               fontsize=8, bbox_to_anchor=(0.5, 0.012))
    fig.text(0.5, -0.004,
             "LR and DM are pure failure endpoints with death censored, so these curves are not "
             "competing-risk adjusted:\nwhere deaths are common, Kaplan–Meier overstates the "
             "probability of failure.",
             ha="center", va="bottom", fontsize=7.5, color=INK2)
    fig.suptitle(title or f"{tgt}: Kaplan–Meier curves by predicted-risk tertile, per model",
                 x=0.02, ha="left", fontsize=12, color=INK)
    fig.tight_layout(rect=[0, 0.05, 0.97, 0.96])
    fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)


def main(tag: str | None = None, run: str | None = None, folder: str | None = None):
    global OUT
    if folder:
        OUT = Path(folder)
    elif run:
        OUT = results_dir("results_external", run)
    elif OUT is None:
        OUT = latest_results_dir("results_external")
    tag = tag or newest_tag()
    rows = pd.read_csv(OUT / f"external_results_{tag}.csv")
    comps = pd.read_csv(OUT / f"external_comparisons_{tag}.csv")
    km = pd.read_csv(OUT / f"external_km_tertiles_{tag}.csv")
    src, tgt = rows.source.iloc[0], rows.target.iloc[0]
    model, train_on, spec = rows.imaging_source.iloc[0], rows.train_on.iloc[0], rows.spec.iloc[0]
    figs = OUT / "figures"; figs.mkdir(parents=True, exist_ok=True)
    pred = lambda ep, arm: (OUT / "predictions" /
                            f"{src}_to_{tgt}_{ep}_{arm}_{model}_{train_on}_{spec}.npz")
    print(f"data version: {DATA_VERSION}   features: {FEATURE_VERSION}   run: {tag}")
    fig_forest(rows, figs / f"ext_forest_{tag}.png", src, tgt)
    made = [f"ext_forest_{tag}.png"]
    if fig_delta(comps, figs / f"ext_delta_{tag}.png", src, tgt):
        made.append(f"ext_delta_{tag}.png")
    fig_km(km, pred, figs / f"ext_km_{tag}.png", tgt)
    made.append(f"ext_km_{tag}.png")
    print(f"figures -> {figs}")
    for m in made:
        print("   ", m)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=None, help="analysis tag inside the run folder; default = newest")
    ap.add_argument("--run", default=None, help="run id YYYYMMDD_HHMMSS; default = newest run")
    ap.add_argument("--dir", default=None, help="any external-results folder")
    a = ap.parse_args()
    main(a.tag, a.run, a.dir)
