"""Unified A1 tables and figures.

Palette: dataviz reference categorical slots 1/2/3 (blue, orange, aqua). These are
the documented all-pairs-validated triple -- worst CVD dE 9.2 (target >=8), worst
normal-vision dE 24.0 (floor >=15) in light mode. Aqua sits at 2.74:1 against the
light surface, below the 3:1 contrast line, so the relief rule applies: every
series is direct-labelled and a table view ships alongside. Distinct markers give
a second, non-colour encoding that survives greyscale print.

Static figures for a print journal, so the skill's hover layer does not apply; the
markdown tables are the table view.

    python make_report.py
"""
from __future__ import annotations

import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd

from cohort import (DATA_VERSION, FEATURE_VERSION, IMAGING, IMAGING_LABEL, IMAGING_SHORT,
                    NSTAGE, latest_results_dir, results_dir)

# set in main(): the run folder to report on (newest by default)
RESULTS = CELLS = REPORT = FIGS = None

# the imaging arm is ctfm_st, or segres_ft when A1_IMAGING=segres_ft
ARMS = ["clinical_only", IMAGING, "combined_A"]
LABEL = {"clinical_only": "Clinical only", IMAGING: IMAGING_LABEL[IMAGING],
         "combined_A": f"Clinical + {IMAGING_SHORT[IMAGING]}"}
COLOR = {"clinical_only": "#2a78d6", IMAGING: "#eb6834", "combined_A": "#1baf7a"}
MARKER = {"clinical_only": "o", IMAGING: "s", "combined_A": "D"}
ENDPOINTS = ["OS", "RFS", "LR", "DM"]
COHORTS = ["Radcure", "Maastro"]

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
    "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.6,
    "legend.frameon": False, "savefig.facecolor": SURFACE,
})


def load_all(spec="primary"):
    rf = sorted(glob.glob(str(CELLS / f"rows_*_{spec}.csv")))
    cf = sorted(glob.glob(str(CELLS / f"comps_*_{spec}.csv")))
    if not rf:
        raise FileNotFoundError(f"no cell results in {CELLS} -- run run_a1.py first")
    def read_nonempty(files):
        out = []
        for f in files:
            try:
                out.append(pd.read_csv(f))
            except pd.errors.EmptyDataError:   # e.g. few-shot: nothing to compare -> empty file
                continue
        return out
    rows = pd.concat(read_nonempty(rf), ignore_index=True)
    cl = read_nonempty(cf)
    comps = pd.concat(cl, ignore_index=True) if cl else pd.DataFrame()
    n_cells = rows[["cohort", "endpoint"]].drop_duplicates().shape[0]
    n_few = rows[rows.regime.isin(["few_shot", "few_shot_pooled"])][["cohort", "endpoint"]].drop_duplicates().shape[0]
    print(f"loaded {n_cells} cells ({rows.cohort.nunique()} cohorts, {rows.endpoint.nunique()} "
          f"endpoints); few-shot results present for {n_few} of them")
    return rows, comps


def _md(df: pd.DataFrame, title: str, note: str) -> str:
    """Markdown table without the optional `tabulate` dependency."""
    cols = list(df.columns)
    w = [max([len(str(c))] + [len(str(v)) for v in df[c]]) for c in cols]
    head = "| " + " | ".join(str(c).ljust(w[i]) for i, c in enumerate(cols)) + " |"
    rule = "|" + "|".join("-" * (x + 2) for x in w) + "|"
    body = "\n".join("| " + " | ".join(str(r[c]).ljust(w[i]) for i, c in enumerate(cols))
                     + " |" for _, r in df.iterrows())
    return f"## {title}\n\n{note}\n\n{head}\n{rule}\n{body}\n"


def full_shot_table(rows):
    f = rows[rows.regime == "full_shot"]
    out = []
    for co in COHORTS:
        for ep in ENDPOINTS:
            for arm in ARMS:
                r = f[(f.cohort == co) & (f.endpoint == ep) & (f.arm == arm)]
                if r.empty:
                    continue
                r = r.iloc[0]
                out.append({"Cohort": co, "Endpoint": ep, "Model": LABEL[arm],
                            "N train": int(r.n_train),
                            "C-index (95% CI)":
                                f"{r.cindex:.3f} ({r.cindex_lo:.3f}–{r.cindex_hi:.3f})",
                            "IBS (95% CI)":
                                f"{r.ibs:.3f} ({r.ibs_lo:.3f}–{r.ibs_hi:.3f})",
                            "EPV": f"{r.epv:.1f}"})
    return pd.DataFrame(out)


def few_shot_table(rows):
    """mean +/- SD across seeds, with the seed-pooled bootstrap CI beneath it."""
    few = rows[rows.regime == "few_shot"]
    pooled = rows[rows.regime == "few_shot_pooled"]
    out = []
    for co in COHORTS:
        sizes = sorted(few[few.cohort == co].n_train.unique())
        for ep in ENDPOINTS:
            if few[(few.cohort == co) & (few.endpoint == ep)].empty:
                continue                     # cell not run yet (partial report)
            for arm in ARMS:
                rec = {"Cohort": co, "Endpoint": ep, "Model": LABEL[arm]}
                for n in sizes:
                    g = few[(few.cohort == co) & (few.endpoint == ep) &
                            (few.arm == arm) & (few.n_train == n)]
                    p = pooled[(pooled.cohort == co) & (pooled.endpoint == ep) &
                               (pooled.arm == arm) & (pooled.n_train == n)]
                    if len(g):
                        s = f"{g.cindex.mean():.3f}±{g.cindex.std():.3f}"
                        if len(p):
                            s += f" [{p.iloc[0].cindex_lo:.3f}–{p.iloc[0].cindex_hi:.3f}]"
                    else:
                        s = "—"
                    rec[f"n={n}"] = s
                out.append(rec)
    return pd.DataFrame(out)


def dl_score_table(comps):
    """All the evidence on the DL score's contribution, one row per cohort x endpoint."""
    if comps.empty:
        return pd.DataFrame()
    def pick(co, ep, metric, a=None, b=None):
        g = comps[(comps.cohort == co) & (comps.endpoint == ep) & (comps.metric == metric)]
        if "regime" in g.columns:
            g = g[g.regime.fillna("full_shot") == "full_shot"]
        if a is not None:
            g = g[g.model_a == a]
        if b is not None:
            g = g[g.model_b == b]
        return None if g.empty else g.iloc[0]
    out = []
    for co in COHORTS:
        for ep in ENDPOINTS:
            w = pick(co, ep, "dl_wald_train")
            t = pick(co, ep, "dl_independent_test")
            l = pick(co, ep, "LRT")
            d = pick(co, ep, "cindex", "combined_A", "clinical_only")
            if all(x is None for x in (w, t, l, d)):
                continue
            pen = (w is not None and bool(w.get("penalized", False)))
            out.append({
                "Cohort": co, "Endpoint": ep,
                "DL HR/SD in combined (train)":
                    "—" if w is None else f"{w.delta:.2f} ({w.lo:.2f}–{w.hi:.2f})"
                                          + (" †" if pen else ""),
                "p (train)": "—" if w is None else f"{w.p:.2g}" + (" †" if pen else ""),
                "LRT p": "—" if l is None else f"{l.p:.2g}",
                "DL HR/SD re-tested (held-out)":
                    "—" if t is None else f"{t.delta:.2f} ({t.lo:.2f}–{t.hi:.2f})",
                "p (held-out)": "—" if t is None else f"{t.p:.2g}",
                "ΔC vs clinical": "—" if d is None else f"{d.delta:+.3f} ({d.lo:+.3f}–{d.hi:+.3f})",
                "p (ΔC)": "—" if d is None else f"{d.p:.3f}",
            })
    return pd.DataFrame(out)


def dl_score_by_shot_table(rows):
    """DL-score contribution at every training-set size (few-shot seeds + full-shot)."""
    if "dl_test_p" not in rows.columns:
        return pd.DataFrame()
    g = rows[(rows.arm == "combined_A") & rows.regime.isin(["few_shot", "full_shot"])]
    out = []
    for co in COHORTS:
        for ep in ENDPOINTS:
            s = g[(g.cohort == co) & (g.endpoint == ep)]
            for n in sorted(s.n_train.unique()):
                x = s[s.n_train == n]
                valid = x[x.dl_penalized == False]                       # noqa: E712
                tst = x[x.dl_test_p.notna()]
                out.append({
                    "Cohort": co, "Endpoint": ep, "N train": int(n),
                    "regime": "full" if (x.regime == "full_shot").all() else "few",
                    "DL HR/SD in combined (train, mean)": f"{x.dl_hr.mean():.2f}" if x.dl_hr.notna().any() else "—",
                    "train p<0.05": (f"{int((valid.dl_p < 0.05).sum())}/{len(valid)} unpenalized"
                                     if len(valid) else f"n/a (all {len(x)} penalized)"),
                    "DL HR/SD re-tested (held-out, mean)": f"{tst.dl_test_hr.mean():.2f}" if len(tst) else "—",
                    "held-out p<0.05": f"{int((tst.dl_test_p < 0.05).sum())}/{len(tst)}",
                })
    return pd.DataFrame(out)


def fig_forest(rows, metric, fname, xlabel):
    f = rows[rows.regime == "full_shot"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.6), sharex=True)
    for ax, co in zip(axes, COHORTS):
        ticks, labels, y = [], [], 0.0
        for ep in ENDPOINTS:
            start = y
            for arm in ARMS:
                r = f[(f.cohort == co) & (f.endpoint == ep) & (f.arm == arm)]
                if not r.empty:
                    r = r.iloc[0]
                    p, lo, hi = r[metric], r[f"{metric}_lo"], r[f"{metric}_hi"]
                    if np.isfinite([p, lo, hi]).all():
                        ax.plot([lo, hi], [y, y], color=COLOR[arm], lw=2,
                                solid_capstyle="round", zorder=2)
                        ax.plot([p], [y], marker=MARKER[arm], ms=8, color=COLOR[arm],
                                mec=SURFACE, mew=2, zorder=3,
                                label=LABEL[arm] if ep == ENDPOINTS[0] else None)
                        ax.annotate(f"{p:.3f}", (hi, y), xytext=(6, 0),
                                    textcoords="offset points", va="center",
                                    fontsize=7.5, color=INK2)
                y += 1
            ticks.append((start + y - 1) / 2); labels.append(ep); y += 0.8
        ax.set_yticks(ticks); ax.set_yticklabels(labels, fontsize=9, color=INK)
        ax.invert_yaxis(); ax.grid(axis="y", visible=False)
        ax.set_title(co, fontsize=10, color=INK, pad=8, loc="left")
        ax.set_xlabel(xlabel)
        if metric == "cindex":
            ax.axvline(0.5, color=MUTED, lw=0.8, ls=(0, (3, 3)), zorder=1)
    g = f[np.isfinite(f[f"{metric}_hi"])]
    if len(g):
        lo_x, hi_x = g[f"{metric}_lo"].min(), g[f"{metric}_hi"].max()
        pad = (hi_x - lo_x) * 0.18
        for ax in axes:
            ax.set_xlim(min(lo_x - pad * 0.25, 0.48 if metric == "cindex" else lo_x),
                        hi_x + pad)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f"Full-shot {xlabel} with 95% bootstrap CI", x=0.02, ha="left",
                 fontsize=12, color=INK)
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])
    fig.savefig(FIGS / fname, dpi=200, bbox_inches="tight"); plt.close(fig)


def fig_curves(rows, fname="fig_fewshot_curves.png"):
    few, full = rows[rows.regime == "few_shot"], rows[rows.regime == "full_shot"]
    fig, axes = plt.subplots(len(ENDPOINTS), 2, figsize=(9.5, 11), sharey=True)
    for i, ep in enumerate(ENDPOINTS):
        for j, co in enumerate(COHORTS):
            ax = axes[i, j]
            sizes = sorted(few[few.cohort == co].n_train.unique())
            for arm in ARMS:
                m, s = [], []
                for n in sizes:
                    g = few[(few.cohort == co) & (few.endpoint == ep) &
                            (few.arm == arm) & (few.n_train == n)]
                    m.append(g.cindex.mean() if len(g) else np.nan)
                    s.append(g.cindex.std() if len(g) else np.nan)
                m, s = np.array(m, float), np.array(s, float)
                ax.fill_between(sizes, m - s, m + s, color=COLOR[arm], alpha=0.13, lw=0)
                ax.plot(sizes, m, color=COLOR[arm], lw=2, marker=MARKER[arm], ms=6,
                        mec=SURFACE, mew=1.5,
                        label=LABEL[arm] if (i == 0 and j == 0) else None)
                fr = full[(full.cohort == co) & (full.endpoint == ep) & (full.arm == arm)]
                if not fr.empty and np.isfinite(fr.iloc[0].cindex):
                    ax.axhline(fr.iloc[0].cindex, color=COLOR[arm], lw=0.9,
                               ls=(0, (2, 3)), alpha=0.55, zorder=1)
            ax.axhline(0.5, color=MUTED, lw=0.8, ls=(0, (3, 3)), zorder=0)
            ax.set_xscale("log"); ax.set_xticks(sizes)
            ax.set_xticklabels([str(n) for n in sizes], fontsize=7, rotation=45)
            ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
            if i == 0:
                ax.set_title(co, fontsize=10, color=INK, pad=8)
            if j == 0:
                ax.set_ylabel(f"{ep}\nC-index", fontsize=9)
            if i == len(ENDPOINTS) - 1:
                ax.set_xlabel("training samples")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle("Few-shot learning curves (mean ± SD over 5 seeds); "
                 "dashed lines = full-shot", x=0.02, ha="left", fontsize=12, color=INK)
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    fig.savefig(FIGS / fname, dpi=200, bbox_inches="tight"); plt.close(fig)


def fig_paired(comps, fname="fig_paired_delta.png"):
    if comps.empty:
        return
    d = comps[(comps.metric == "cindex") & (comps.model_b == "clinical_only")]
    if d.empty:
        return
    arms = [a for a in [IMAGING, "combined_A"] if a in set(d.model_a)]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), sharex=True)
    for ax, co in zip(axes, COHORTS):
        ticks, labels, y = [], [], 0.0
        for ep in ENDPOINTS:
            start = y
            for arm in arms:
                r = d[(d.cohort == co) & (d.endpoint == ep) & (d.model_a == arm)]
                if not r.empty:
                    r = r.iloc[0]
                    sig = np.isfinite(r.p) and r.p < 0.05
                    ax.plot([r.lo, r.hi], [y, y], color=COLOR[arm], lw=2,
                            solid_capstyle="round", zorder=2)
                    ax.plot([r.delta], [y], marker=MARKER[arm], ms=8, color=COLOR[arm],
                            mec=SURFACE, mew=2, zorder=3,
                            label=f"{LABEL[arm]} − Clinical" if ep == ENDPOINTS[0] else None)
                    ax.annotate(f"{r.delta:+.3f}" + ("*" if sig else ""), (r.hi, y),
                                xytext=(6, 0), textcoords="offset points", va="center",
                                fontsize=7.5, color=INK if sig else INK2)
                y += 1
            ticks.append((start + y - 1) / 2); labels.append(ep); y += 0.8
        ax.axvline(0, color=INK, lw=1.0, zorder=1)
        ax.set_yticks(ticks); ax.set_yticklabels(labels, fontsize=9, color=INK)
        ax.invert_yaxis(); ax.grid(axis="y", visible=False)
        ax.set_title(co, fontsize=10, color=INK, pad=8, loc="left")
        ax.set_xlabel("Δ C-index vs clinical-only")
    g = d[np.isfinite(d.hi)]
    if len(g):
        pad = (g.hi.max() - g.lo.min()) * 0.18
        for ax in axes:
            ax.set_xlim(g.lo.min() - pad * 0.25, g.hi.max() + pad)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Incremental value over the clinical baseline "
                 "(paired bootstrap; * = p < 0.05)", x=0.02, ha="left",
                 fontsize=12, color=INK)
    fig.tight_layout(rect=[0, 0.07, 1, 0.94])
    fig.savefig(FIGS / fname, dpi=200, bbox_inches="tight"); plt.close(fig)


_N_NOTE = ("N stage harmonised to AJCC (p16-specific) in both cohorts"
           if NSTAGE == "ajcc8" else
           "N stage as each cohort recorded it -- different AJCC versions; NOT "
           "on a common scale")
SPEC_NOTE = {"primary": f"Clinical model: age, sex, HPV, smoking, T stage, N stage. {_N_NOTE}.",
             "ajcc": "Clinical model: age, sex, HPV, smoking, AJCC stage group "
                     "(sensitivity analysis: overall stage instead of T and N)."}


def report_dir(results: Path, spec: str = "primary") -> Path:
    """_report/ for the primary clinical model, _report_<spec>/ for a sensitivity spec."""
    return results / ("_report" if spec == "primary" else f"_report_{spec}")


def main(run: str | None = None, folder: str | None = None, spec: str = "primary"):
    global RESULTS, CELLS, REPORT, FIGS
    RESULTS = (Path(folder) if folder else
               results_dir("results", run) if run else latest_results_dir("results"))
    CELLS, REPORT = RESULTS / "_cells", report_dir(RESULTS, spec)
    FIGS = REPORT / "figures"
    FIGS.mkdir(parents=True, exist_ok=True)
    print(f"data version: {DATA_VERSION}   imaging: {IMAGING_LABEL[IMAGING]}   "
          f"features: {FEATURE_VERSION}   N stage: {NSTAGE}   reading {CELLS}")
    print(f"clinical spec: {spec}   -> {REPORT}")
    rows, comps = load_all(spec)
    rows.to_csv(REPORT / "a1_all_models.csv", index=False)
    if not comps.empty:
        comps.to_csv(REPORT / "a1_comparisons.csv", index=False)

    has_few = (rows.regime == "few_shot").any()
    ft = full_shot_table(rows)
    ft.to_csv(REPORT / "a1_fullshot_table.csv", index=False)
    (REPORT / "a1_fullshot_table.md").write_text(_md(
        ft, "Full-shot discrimination and calibration",
        "Trained on the complete training split, evaluated on the held-out test set. "
        "95% CIs from 2,000 event-stratified bootstrap resamples of the test set. "
        + SPEC_NOTE.get(spec, "")))
    if has_few:
        wt = few_shot_table(rows)
        wt.to_csv(REPORT / "a1_fewshot_table.csv", index=False)
        (REPORT / "a1_fewshot_table.md").write_text(_md(
            wt, "Few-shot C-index",
            "mean ± SD across 5 seeds, with [seed-pooled bootstrap 95% CI]. Same held-out "
            "test set as full-shot."))
    else:                                   # full-shot-only report: drop stale few-shot files
        for f in ["a1_fewshot_table.csv", "a1_fewshot_table.md"]:
            (REPORT / f).unlink(missing_ok=True)
        (FIGS / "fig_fewshot_curves.png").unlink(missing_ok=True)
        print("few-shot results not present yet -> full-shot report only")

    dl = dl_score_table(comps)
    if not dl.empty:
        dl.to_csv(REPORT / "a1_dl_score_table.csv", index=False)
        (REPORT / "a1_dl_score_table.md").write_text(_md(
            dl, f"Contribution of the DL ({IMAGING_SHORT[IMAGING]}) score",
            "HR per SD of the DL score. *Train*: Wald test inside the fitted combined model "
            "-- optimistic, because the score is itself estimated from the training data. "
            "*Held-out*: frozen clinical and DL scores re-tested together on test patients "
            "neither was fitted on -- the cleaner test. LRT: clinical vs clinical + DL. "
            "ΔC: paired-bootstrap C-index gain on the held-out set. "
            "† penalized fit: train p-value not valid."))

    bs = dl_score_by_shot_table(rows)
    if not bs.empty:
        bs.to_csv(REPORT / "a1_dl_score_by_shot.csv", index=False)
        (REPORT / "a1_dl_score_by_shot.md").write_text(_md(
            bs, "DL score contribution by training-set size",
            "Combined model only (the one model with the DL score as an input). Few-shot "
            "rows summarise the 5 seeds; full-shot is a single fit. *Train* p-values count "
            "only unpenalized fits -- penalized fits do not give valid p-values. The "
            "*held-out re-test* is valid at every size and is the one to read for few-shot."))

    fig_forest(rows, "cindex", "fig_forest_cindex.png", "C-index")
    fig_forest(rows, "ibs", "fig_forest_ibs.png", "Integrated Brier Score")
    if has_few:
        fig_curves(rows)
    fig_paired(comps)
    print(f"\ntables  -> {REPORT}")
    print(f"figures -> {FIGS}")
    for f in sorted(FIGS.glob("*.png")):
        print("   ", f.name)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None, help="run id YYYYMMDD_HHMMSS; default = newest run")
    ap.add_argument("--dir", default=None, help="any results folder, e.g. the legacy results/")
    ap.add_argument("--spec", choices=["primary", "ajcc"], default="primary",
                    help="clinical model: primary = T+N, ajcc = AJCC stage (-> _report_ajcc/)")
    a = ap.parse_args()
    main(a.run, a.dir, a.spec)
