"""Figure 4, rebuilt: subgroup C-index with 95% CI, from run_subgroup.py's long CSV.

4 endpoints x 2 cohorts; within each panel one row per subgroup, one mark per model.
The submitted figure used bars; a C-index has no meaningful zero (0.5 is chance), so
bars from 0 exaggerate differences and leave no room for the CI R3 asked for. Dot and
whisker with a chance line reads the same way and carries the interval.

Colours and markers are the ones make_figures_s4 uses, so a model keeps its identity
across every revision figure.

    python make_figure_s5.py --long .../table_s5_subgroups_long.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from make_figures_s4 import STYLE, _models
from make_report import INK, INK2, MUTED, SURFACE

ENDPOINTS = ["OS", "RFS", "LR", "DM"]
COHORTS = ["Radcure", "Maastro"]


def main(long_csv: str, out: str | None = None, clinical: bool = True):
    long_csv = Path(long_csv)
    df = pd.read_csv(long_csv)
    out = Path(out) if out else long_csv.parent / "figures"
    out.mkdir(parents=True, exist_ok=True)
    models = _models(df, clinical)
    order = [s for s in df.subgroup.unique()]          # file order = Figure 4 order
    print(f"figure 4: models {', '.join(models)}")

    fig, axes = plt.subplots(len(ENDPOINTS), len(COHORTS), figsize=(11.5, 15.5), sharex=True)
    for i, ep in enumerate(ENDPOINTS):
        for j, co in enumerate(COHORTS):
            ax = axes[i, j]
            sub = df[(df.cohort == co) & (df.endpoint == ep)]
            present = [s for s in order if not sub[sub.subgroup == s].empty]
            ticks, labels, y = [], [], 0.0
            for s in present:
                start = y
                g = sub[sub.subgroup == s]
                for m in models:
                    r = g[g.model == m]
                    if not r.empty:
                        r = r.iloc[0]
                        col, mk, _, _ = STYLE[m]
                        if np.isfinite(r.lo):
                            ax.plot([r.lo, r.hi], [y, y], color=col, lw=1.8,
                                    solid_capstyle="round", zorder=2)
                        ax.plot([r.cindex], [y], marker=mk, ms=6.5, color=col, mec=SURFACE,
                                mew=1.5, ls="none", zorder=3,
                                label=m if (i == 0 and j == 0 and s == present[0]) else None)
                    y += 1
                ticks.append((start + y - 1) / 2)
                n = int(g.iloc[0].n)
                labels.append(f"{s} ({n})")
                y += 0.8
            ax.axvline(0.5, color=MUTED, lw=0.9, ls=(0, (3, 2)), zorder=1)
            ax.set_yticks(ticks)
            ax.set_yticklabels(labels, fontsize=7.5, color=INK)
            ax.invert_yaxis()
            ax.grid(axis="y", visible=False)
            ax.set_title(f"{ep} — {co}", fontsize=10, color=INK, pad=6, loc="left")
            if i == len(ENDPOINTS) - 1:
                ax.set_xlabel("C-index (95% CI); subgroup size in brackets")
    lo = df.lo[df.model.isin(models)].min(skipna=True)
    hi = df.hi[df.model.isin(models)].max(skipna=True)
    for ax in axes.flat:
        ax.set_xlim(max(0.0, min(0.3, lo - 0.03)), min(1.0, max(0.9, hi + 0.03)))
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper left", ncol=len(l), fontsize=8.5, bbox_to_anchor=(0.012, 0.972))
    fig.suptitle("Subgroup discrimination on the held-out test sets (95% CI)",
                 x=0.02, ha="left", fontsize=12, color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.952])
    p = out / "fig4_subgroups_95ci.png"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"   {p}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--long", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-clinical", action="store_true")
    a = ap.parse_args()
    main(a.long, a.out, clinical=not a.no_clinical)
