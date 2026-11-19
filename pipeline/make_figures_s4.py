"""Figures 2 and 3 of the manuscript, rebuilt with 95% confidence intervals.

Both are drawn from the long CSV that make_table_s4.py writes, so the figures and the
revised Table S4 always show the same numbers.

  Figure 2  full-shot C-index per model with its 95% CI (dot and whisker). The submitted
            figure used bars with +/- SD across seeds; a C-index has no meaningful zero,
            so bars from 0 exaggerate differences.
  Figure 3  C-index against training-set size, mean over seeds with a shaded 95% CI band
            (the submitted figure shaded +/- SD).

Colour. Overlapping lines need every PAIR of colours to stay distinguishable, and only the
first three categorical slots pass that test (dataviz palette validator, all pairs, light):
blue, orange, aqua. CT-FM-ST keeps the orange it has in every revision figure (make_report);
CT-FM-LP takes blue and the from-scratch SegRes model aqua. SPECTRE (OS only) is drawn in
muted grey, dashed, and the clinical-only model as a dark dotted reference line -- neither
competes with the categorical slots. Every series also has its own marker and line style,
so identity never rests on colour alone; aqua sits below 3:1 contrast, and Table S4 is the
table view that relief requires.

    python make_figures_s4.py --long results_.../_report/table_s4_cindex_long.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from make_report import INK, INK2, MUTED, SURFACE   # shared style; importing applies it

ENDPOINTS = ["OS", "RFS", "LR", "DM"]
COHORTS = ["Radcure", "Maastro"]
ORANGE, BLUE, AQUA = "#eb6834", "#2a78d6", "#1baf7a"
# model -> (colour, marker, line style, CI band). Order = legend and dodge order.
STYLE = {
    "Clinical only": (INK, "o", (0, (1, 1.6)), False),
    "CT-FM-ST": (ORANGE, "s", "-", True),
    "CT-FM-LP": (BLUE, "o", "-", True),
    "SegRes-E2E": (AQUA, "D", "-", True),
    "SegRes-FT": (AQUA, "D", "-", True),
    "SPECTRE": (MUTED, "^", (0, (4, 2)), True),
}


def _models(df: pd.DataFrame, clinical: bool) -> list[str]:
    have = set(df.model)
    models = [m for m in STYLE if m in have and (clinical or m != "Clinical only")]
    if "SegRes-E2E" in models and "SegRes-FT" in models:   # one aqua slot for one model
        print("both SegRes-E2E and SegRes-FT present -- drawing SegRes-E2E (the paper's model)")
        models.remove("SegRes-FT")
    return models


def _labelled_ticks(xs, min_frac=0.06):
    """Training sizes to label on a log axis: always the last (full shot), then leftwards
    every size at least `min_frac` of the axis away from the last labelled one. The rest
    keep an unlabelled tick (e.g. Radcure 752 sits too close to 940 to carry a label)."""
    xs = sorted(xs)
    span = np.log10(xs[-1] / xs[0]) or 1.0
    keep = [xs[-1]]
    for x in reversed(xs[:-1]):
        if np.log10(keep[-1] / x) / span >= min_frac:
            keep.append(x)
    return sorted(keep)


def fig_fullshot(df, models, path):
    full = df[df.train_size == "full"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.4), sharex=True)
    for ax, co in zip(axes, COHORTS):
        ticks, y = [], 0.0
        n_te = None
        for ep in ENDPOINTS:
            start = y
            for m in models:
                r = full[(full.cohort == co) & (full.endpoint == ep) & (full.model == m)]
                if not r.empty:
                    r = r.iloc[0]
                    col, mk, _, _ = STYLE[m]
                    ax.plot([r.lo, r.hi], [y, y], color=col, lw=2, solid_capstyle="round", zorder=2)
                    ax.plot([r.value], [y], marker=mk, ms=7.5, color=col, mec=SURFACE, mew=1.8,
                            ls="none", zorder=3, label=m if ep == ENDPOINTS[0] else None)
                    n_te = int(r.n_test)
                y += 1
            ticks.append((start + y - 1) / 2)
            y += 0.9
        ax.axvline(0.5, color=MUTED, lw=0.9, ls=(0, (3, 2)), zorder=1)
        ax.set_yticks(ticks)
        ax.set_yticklabels(ENDPOINTS, fontsize=9, color=INK)
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
        ax.set_title(f"{co}" + (f"  (test n = {n_te})" if n_te else ""), fontsize=10, color=INK,
                     pad=8, loc="left")
        ax.set_xlabel("C-index (95% CI)")
    lo = full.lo[full.model.isin(models)].min()
    hi = full.hi[full.model.isin(models)].max()
    for ax in axes:
        ax.set_xlim(min(0.45, lo - 0.02), max(0.95, hi + 0.02))
    axes[0].annotate("chance", (0.5, 1.0), xycoords=("data", "axes fraction"), xytext=(3, -10),
                     textcoords="offset points", fontsize=7.5, color=MUTED)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(l), fontsize=8.5, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Full-shot discrimination on the held-out test sets (95% CI)",
                 x=0.02, ha="left", fontsize=12, color=INK)
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_curves(df, models, path, n_full):
    fig, axes = plt.subplots(len(ENDPOINTS), len(COHORTS), figsize=(10.5, 13.2), sharey=True)
    seen = set()          # legend entry per model, wherever it first appears
    for i, ep in enumerate(ENDPOINTS):
        for j, co in enumerate(COHORTS):
            ax = axes[i, j]
            for m in models:
                g = df[(df.cohort == co) & (df.endpoint == ep) & (df.model == m)].copy()
                if g.empty:
                    continue
                g["x"] = [n_full.get(co, np.nan) if s == "full" else int(s) for s in g.train_size]
                g = g.dropna(subset=["x"]).sort_values("x")
                col, mk, ls, band = STYLE[m]
                if band and len(g) > 1:
                    ax.fill_between(g.x, g.lo, g.hi, color=col, alpha=0.12 if col != MUTED else 0.10,
                                    lw=0, zorder=1)
                ax.plot(g.x, g.value, color=col, lw=2 if m != "Clinical only" else 1.6, ls=ls,
                        marker=mk, ms=5.5, mec=SURFACE, mew=1.2, zorder=3,
                        label=m if m not in seen else None)
                seen.add(m)
            ax.axhline(0.5, color=MUTED, lw=0.9, ls=(0, (3, 2)), zorder=0)
            ax.set_xscale("log")
            xs = sorted({int(s) for s in df[df.cohort == co].train_size if s != "full"}
                        | ({int(n_full[co])} if co in n_full else set()))
            shown = _labelled_ticks(xs)
            ax.xaxis.set_major_locator(mticker.FixedLocator(shown))
            ax.xaxis.set_major_formatter(mticker.FixedFormatter([str(x) for x in shown]))
            ax.xaxis.set_minor_locator(mticker.FixedLocator([x for x in xs if x not in shown]))
            ax.xaxis.set_minor_formatter(mticker.NullFormatter())
            ax.tick_params(axis="x", labelsize=7.5)
            ax.set_title(f"{ep} — {co}", fontsize=10, color=INK, pad=6, loc="left")
            if j == 0:
                ax.set_ylabel("C-index")
            if i == len(ENDPOINTS) - 1:
                ax.set_xlabel("Training patients (log scale; last point = full training split)")
    lo = df.lo[df.model.isin(models)].min()
    hi = df.hi[df.model.isin(models)].max()
    axes[0, 0].set_ylim(max(0.0, min(0.4, lo - 0.02)), min(1.0, max(0.85, hi + 0.02)))
    h, l = [], []
    for ax in axes.flat:                               # a model may first appear in any panel
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            h.append(hh); l.append(ll)
    # legend at the top, where a reader starts -- the figure is tall
    fig.legend(h, l, loc="upper left", ncol=len(l), fontsize=8.5, bbox_to_anchor=(0.012, 0.968))
    fig.suptitle("C-index by training-set size (mean over 5 seeds; shaded = 95% CI)",
                 x=0.02, ha="left", fontsize=12, color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.945])
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main(long_csv: str, out: str | None = None, clinical: bool = True, n_full: dict | None = None):
    long_csv = Path(long_csv)
    df = pd.read_csv(long_csv, dtype={"train_size": str})
    out = Path(out) if out else long_csv.parent / "figures"
    out.mkdir(parents=True, exist_ok=True)
    if n_full is None:   # full-shot training size per cohort, stored by make_table_s4.py
        n_full = ({r.cohort: int(r.n_train_full) for r in df.drop_duplicates("cohort").itertuples()}
                  if "n_train_full" in df.columns else {"Radcure": 940, "Maastro": 121})
    models = _models(df, clinical)
    print(f"figures: models {', '.join(models)}")
    fig_fullshot(df, models, out / "fig2_fullshot_cindex_95ci.png")
    fig_curves(df, models, out / "fig3_fewshot_curves_95ci.png", n_full)
    for f in ("fig2_fullshot_cindex_95ci.png", "fig3_fewshot_curves_95ci.png"):
        print(f"   {out / f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--long", required=True, help="table_s4_cindex_long.csv from make_table_s4.py")
    ap.add_argument("--out", default=None, help="output folder (default: figures/ next to the CSV)")
    ap.add_argument("--no-clinical", action="store_true", help="leave out the clinical reference")
    a = ap.parse_args()
    main(a.long, a.out, clinical=not a.no_clinical)
