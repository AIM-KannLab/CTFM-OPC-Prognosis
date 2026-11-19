"""Merged report of a run: OS/RFS from the original-data runs, LR/DM from the corrected-data runs.

Reads the per-data-version reports run_pipeline.sh made (report_original/, report_corrected/: make_table_s4,
run_subgroup) and the two external runs, and writes <root>/report/:
  table_s4.md / table_s4_cindex_long.csv            internal, every model and size, C-index (95% CI)
  figures/fig2_fullshot_cindex_95ci.png, fig2_fullshot_bars_95ci.png, fig3_fewshot_curves_95ci.png
  table_s5_subgroups.md / figures/fig4_subgroups      subgroups, full shot
  external/  external_summary_*.md + ext_forest / ext_delta / ext_km figures (CT-FM-ST, clinical, combined)
  comparison_with_paper.md                                 every Table S4 number of the submitted paper next to this run
Run through runner.py (run_pipeline.sh does):
    python3 runner.py run --data corrected --features st --root <root> -- merged_report.py --root <root>
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from ctfm_config import data_path

HERE = Path(__file__).resolve().parent
# Text export of the submitted manuscript (config.yaml submitted.paper_md, optional): only for
# comparison_with_paper.md; skipped when not configured.
PAPER_MD = data_path("submitted.paper_md")
KEEP = {"original": ["OS", "RFS"], "corrected": ["LR", "DM"]}
EP_ORDER = ["OS", "RFS", "LR", "DM"]
MODELS = ["Clinical only", "Clinical + CT-FM", "CT-FM-ST", "CT-FM-LP", "SegRes-E2E", "SPECTRE"]
TAG = "Radcure_to_Maastro_refit_train_primary"


def merge(root: Path, name: str, col: str = "endpoint") -> pd.DataFrame:
    parts = []
    for d, eps in KEEP.items():
        f = root / f"report_{d}" / name
        if f.exists():
            x = pd.read_csv(f, dtype={"train_size": str} if "table_s4" in name else None)
            parts.append(x[x[col].isin(eps)].assign(data_version=d))
        else:
            print(f"!! missing {f}")
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def table_s4_md(d: pd.DataFrame) -> str:
    sizes = ["10", "24", "50", "100", "full"]
    cell = lambda r, s: (f"{r.value:.3f} ({r.lo:.3f}–{r.hi:.3f})" if np.isfinite(r.lo) else f"{r.value:.3f}") + \
        (f" †{int(r.n_seeds)}" if s != "full" and r.n_seeds < 5 else "")
    L = ["## Table S4 (this run): C-index (95% CI) on the held-out test sets", "",
         "Mean over 5 seeds at each few-shot size; CT-FM-ST / clinical models at full shot are one deterministic fit, "
         "CT-FM-LP / SegRes-E2E / SPECTRE the mean of 5 runs. 95% CI: 2,000 bootstrap resamples of the test patients, "
         "applied jointly to all seeds. OS and RFS: the submitted data; LR and DM: corrected (pure) LR/DM events. "
         "†k = only k of 5 seeds could be fitted.", "",
         "| Endpoint | Cohort | Model | 10 | 24 | 50 | 100 | Full-shot [N] |", "|---|---|---|---|---|---|---|---|"]
    for ep in EP_ORDER:
        for co in ["Radcure", "Maastro"]:
            for m in MODELS:
                g = d[(d.endpoint == ep) & (d.cohort == co) & (d.model == m)].set_index("train_size")
                if g.empty:
                    continue
                vals = [cell(g.loc[s], s) if s in g.index else "—" for s in sizes]
                n_full = int(g.n_train_full.iloc[0]) if "n_train_full" in g else ""
                vals[-1] += f" [{n_full}]"
                L.append(f"| {ep} | {co} | {m} | " + " | ".join(vals) + " |")
    return "\n".join(L) + "\n"


def paper_values() -> pd.DataFrame:
    """Table S4 of the submitted manuscript (mean over seeds), parsed from its text export."""
    lines = [l.strip() for l in PAPER_MD.read_text().splitlines()]
    i = max(k for k, l in enumerate(lines) if l.startswith("Table S4"))
    toks = [t for t in lines[i + 1:i + 700] if t]
    rows, k = [], 0
    while k < len(toks) - 7:
        if toks[k] in EP_ORDER and toks[k + 1] in ("Radcure", "Maastro"):
            vals = toks[k + 3:k + 8]
            if all(re.match(r"^[0-9.]+", v) for v in vals):
                rows.append([toks[k], toks[k + 1], toks[k + 2]] + [float(v.split()[0]) for v in vals])
                k += 8
                continue
        k += 1
    p = pd.DataFrame(rows, columns=["endpoint", "cohort", "model", "10", "24", "50", "100", "full"])
    return p.melt(id_vars=["endpoint", "cohort", "model"], var_name="train_size", value_name="paper")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", required=True)
    a = ap.parse_args()
    root = Path(a.root)
    out = root / "report"; (out / "figures").mkdir(parents=True, exist_ok=True)

    # ---- Table S4 + Figures 2 and 3
    s4 = merge(root, "table_s4_cindex_long.csv")
    s4.to_csv(out / "table_s4_cindex_long.csv", index=False)
    s4.to_csv(out / "table_s4_cindex_long_all_models.csv", index=False)       # name make_figure2_bars reads
    (out / "table_s4.md").write_text(table_s4_md(s4))
    try:
        import make_figures_s4 as F
        style = dict(F.STYLE)
        F.STYLE = {"Clinical only": style.pop("Clinical only"),
                   "Clinical + CT-FM": ("#8f2d6b", "v", (0, (5, 1.5, 1, 1.5)), True), **style}
        F.main(str(out / "table_s4_cindex_long.csv"), str(out / "figures"))
        import make_figure2_bars as B
        B.REPORT = out
        B.main()
    except Exception as e:
        print(f"!! Figures 2/3 failed: {e}")

    # ---- paper vs this run
    try:
        p = paper_values()
        r = s4[["endpoint", "cohort", "model", "train_size", "value", "lo", "hi"]]
        c = p.merge(r, on=["endpoint", "cohort", "model", "train_size"], how="left")
        c["diff"] = c.value - c.paper
        c["paper_inside_ci"] = (c.paper >= c.lo) & (c.paper <= c.hi)
        c.to_csv(out / "comparison_with_paper.csv", index=False)
        L = ["## Submitted Table S4 vs this run", "",
             "Paper = the submitted value (mean over its seeds; CT-FM-ST few-shot used 10 seeds there, 5 here). "
             "LR/DM here use the corrected (pure) events, so they are expected to differ.", "",
             "| Endpoint | Cohort | Model | Size | Paper | This run (95% CI) | Δ |", "|---|---|---|---|---|---|---|"]
        for x in c.itertuples():
            rep = "—" if pd.isna(x.value) else f"{x.value:.3f} ({x.lo:.3f}–{x.hi:.3f})"
            dd = "—" if pd.isna(x.diff) else f"{x.diff:+.3f}"
            L.append(f"| {x.endpoint} | {x.cohort} | {x.model} | {x.train_size} | {x.paper:.3f} | {rep} | {dd} |")
        os_rfs = c[c.endpoint.isin(["OS", "RFS"]) & c.value.notna()]
        L += ["", f"OS/RFS (same data as the paper): {os_rfs.paper_inside_ci.mean():.0%} of the paper's values lie "
              f"inside this run's 95% CI; median |Δ| {os_rfs['diff'].abs().median():.3f}."]
        (out / "comparison_with_paper.md").write_text("\n".join(L) + "\n")
    except Exception as e:                                  # the comparison is a convenience, not the result
        print(f"!! paper comparison skipped: {e}")

    # ---- subgroups (Table S5 / Figure 4)
    s5 = merge(root, "table_s5_subgroups_long.csv")
    if not s5.empty:
        s5.to_csv(out / "table_s5_subgroups_long.csv", index=False)
        try:
            import make_figure_s5 as S5
            S5.main(str(out / "table_s5_subgroups_long.csv"), str(out / "figures"))
        except Exception as e:
            print(f"!! Figure 4 (subgroups) failed: {e}")
        md = [f"## Table S5 (this run): subgroup C-index, full shot", "",
              "Per-version tables with every column: report_original/table_s5_subgroups.md (OS, RFS), "
              "report_corrected/table_s5_subgroups.md (LR, DM)."]
        (out / "table_s5_subgroups.md").write_text("\n".join(md) + "\n")

    # ---- external validation (CT-FM-ST, clinical, Clinical + CT-FM)
    ext = out / "external"; ext.mkdir(exist_ok=True)
    frames = {"results": [], "comparisons": [], "km_tertiles": []}
    for d, eps in KEEP.items():
        hit = sorted(root.glob(f"results_external_{d}_*"))
        if not hit:
            print(f"!! no external run for {d}"); continue
        for kind in frames:
            f = hit[-1] / f"external_{kind}_{TAG}.csv"
            if f.exists():
                x = pd.read_csv(f)
                frames[kind].append(x[x.endpoint.isin(eps)])
    if frames["results"]:
        for kind, xs in frames.items():
            pd.concat(xs, ignore_index=True).to_csv(ext / f"external_{kind}_{TAG}.csv", index=False)
        try:
            import run_external as RE
            rows, comps, kms = (pd.read_csv(ext / f"external_{k}_{TAG}.csv") for k in frames)
            (ext / f"external_summary_{TAG}.md").write_text(
                RE._summary_md(rows, comps, kms, "Radcure", "Maastro", "refit", "train"))
            import make_external_report as MER
            MER.main(TAG, folder=str(ext))
        except Exception as e:
            print(f"!! external summary/figures failed: {e}")
    print(f"report -> {out}")


if __name__ == "__main__":
    main()
