"""Figure 2 as a bar chart, in the layout of the submitted figure, with 95% CIs as the error bars.

Rows OS, LR, DM, RFS; columns Radcure, Maastro (as submitted). Bars = full-shot C-index on the
held-out test set; error bars = 95% CI (2,000 bootstrap resamples of the test patients, applied jointly
to all seeds -- make_table_s4.py), instead of the submitted +/- SD over seeds. Six models: the
clinical model and Clinical + CT-FM are new. Numbers: the merged Table S4 of the revision
(table_s4_cindex_long_all_models.csv: main run + SPECTRE RFS/LR/DM). Colours: those of the revised
Figure 3 (make_figures_s4.py + Clinical + CT-FM in wine).

    python3 make_figure2_bars.py          (merged_report.py calls it with REPORT = <run>/report)
-> <outputs.dir>/<run>_study/report/figures/fig2_fullshot_bars_95ci.{png,pdf}
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402
import pandas as pd                                   # noqa: E402

from make_report import INK, INK2, MUTED, SURFACE     # noqa: E402  (shared style)
from ctfm_config import paper_run                     # noqa: E402

HERE = Path(__file__).resolve().parent
REPORT = paper_run(must_exist=False)[0] / "report"    # newest run, or outputs.paper_run
MODELS = {"Clinical only": INK, "Clinical + CT-FM": "#8f2d6b", "CT-FM-ST": "#eb6834",
          "CT-FM-LP": "#2a78d6", "SegRes-E2E": "#1baf7a", "SPECTRE": "#8a8984"}
ENDPOINTS = ["OS", "LR", "DM", "RFS"]                 # row order of the submitted figure
COHORTS = ["Radcure", "Maastro"]


def main():
    d = pd.read_csv(REPORT / "table_s4_cindex_long_all_models.csv", dtype={"train_size": str})
    d = d[d.train_size == "full"]
    fig, axes = plt.subplots(len(ENDPOINTS), len(COHORTS), figsize=(11, 15.5), sharey=True)
    for i, ep in enumerate(ENDPOINTS):
        for j, co in enumerate(COHORTS):
            ax = axes[i, j]
            g = d[(d.endpoint == ep) & (d.cohort == co)].set_index("model")
            ms = [m for m in MODELS if m in g.index]
            x = range(len(ms))
            v = [g.loc[m, "value"] for m in ms]
            err = [[g.loc[m, "value"] - g.loc[m, "lo"] for m in ms], [g.loc[m, "hi"] - g.loc[m, "value"] for m in ms]]
            ax.bar(x, v, width=0.8, color=[MODELS[m] for m in ms], edgecolor=SURFACE, linewidth=1.5, zorder=2)
            # light halo under the error bar so it stays visible on the dark (black, wine) bars
            ax.errorbar(x, v, yerr=err, fmt="none", ecolor=SURFACE, elinewidth=3.2, capsize=4.5, capthick=3.2, zorder=3)
            ax.errorbar(x, v, yerr=err, fmt="none", ecolor=INK, elinewidth=1.2, capsize=3.5, capthick=1.2, zorder=4)
            for xi, m in zip(x, ms):
                ax.annotate(f"{g.loc[m, 'value']:.3f}", (xi, g.loc[m, "hi"]), xytext=(0, 3),
                            textcoords="offset points", ha="center", va="bottom", fontsize=7.5, color=INK)
            ax.axhline(0.5, color=MUTED, lw=0.9, ls=(0, (3, 2)), zorder=1)
            ax.set_xticks(list(x))
            ax.set_xticklabels(ms, rotation=30, ha="right", fontsize=8.5)
            ax.set_ylim(0, 1.1)
            ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
            ax.grid(axis="x", visible=False)
            n_te, n_ev = int(g.n_test.iloc[0]), int(g.n_events.iloc[0])
            ax.set_title(f"{co} — {ep} (full shot)   test n = {n_te}, {n_ev} events",
                         fontsize=10, color=INK, loc="left")
            if j == 0:
                ax.set_ylabel("C-index")
    axes[0, 0].annotate("chance", (len(MODELS) - 0.6, 0.5), xytext=(0, 3), textcoords="offset points",
                        ha="right", va="bottom", fontsize=7.5, color=MUTED)
    fig.suptitle("Full-shot C-index on the held-out test sets (error bars: 95% confidence interval)",
                 x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    out = REPORT / "figures"
    for ext in ("png", "pdf"):
        fig.savefig(out / f"fig2_fullshot_bars_95ci.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}/fig2_fullshot_bars_95ci.{{png,pdf}}")


if __name__ == "__main__":
    main()
