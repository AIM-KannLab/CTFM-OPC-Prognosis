"""SPECTRE + Cox head: the CT-FM-ST ridge Cox head fitted on the frozen SPECTRE features (overall survival only).

A check of the prediction head: in the main models SPECTRE has a linear head and CT-FM a Cox head. This runs the
CT-FM-ST pipeline unchanged (run_a1.py internal few/full shot with the clinical and combined models; run_external.py
Radcure -> Maastro) with the feature folder pointed at the SPECTRE features (cache/features_spectre_paper/, 1,080
features per scan). The scripts label the imaging arm "ctfm_st"; here it is SPECTRE + Cox head. Its rows of
Supplementary Tables S5-S7 (OS) are then computed with the same methods as the main tables.

run_pipeline.sh runs it after the SPECTRE features exist; by hand:
    python pipeline/run_spectre_cox.py a1 --cohort Radcure --root <run folder>/spectre_cox --run <run id>
    python pipeline/run_spectre_cox.py a1 --cohort Maastro --root <run folder>/spectre_cox --run <run id>
    python pipeline/run_spectre_cox.py external --root <run folder>/spectre_cox --run <run id>
    python pipeline/run_spectre_cox.py rows --root <run folder>/spectre_cox --main <run folder> --out <run folder>/report
-> <out>/spectre_cox_rows.json and spectre_cox_rows.md
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import runner as R                                                  # noqa: E402

R.FEAT["spectre"] = R.SPECTRE_FEAT                                  # the only change to the CT-FM-ST pipeline
LAB, LABC = "SPECTRE + Cox head", "Clinical + SPECTRE + Cox head"


def run_script(root, run, script_args):
    if not (R.SPECTRE_FEAT / "features_radcure.npz").exists():
        raise SystemExit(f"missing {R.SPECTRE_FEAT}/features_radcure.npz -- extract the SPECTRE features first")
    R.cmd_run(argparse.Namespace(data="original", features="spectre", root=root, seeds=5,
                                 script=script_args[:1] + ["--run", run] + script_args[1:]))


# ------------------------------------------------------------------------------------------------ table rows
def _one(folder: Path, pattern: str) -> Path:
    hits = sorted(glob.glob(str(folder / pattern)))
    assert len(hits) == 1, (pattern, hits)
    return Path(hits[0])


def rows(root: Path, main: Path, out: Path):
    import numpy as np
    import pandas as pd
    from bootstrap_ci import bootstrap_ci, cindex, paired_bootstrap
    from ctfm_config import test_bootstrap_b
    B = test_bootstrap_b() or 2000                # the published 2,000 resamples (smoke test: fewer)

    ci = lambda v, lo, hi: f"{v:.3f} ({lo:.3f}–{hi:.3f})"
    dci = lambda d, lo, hi: f"{d:+.3f} ({lo:+.3f} to {hi:+.3f})"
    pf = lambda p: "< 0.001" if p < 0.001 else f"{p:.3f}"

    def cell(path, n_test):                    # Table S5 cell: mean over the subsets, joint bootstrap CI
        if not path.exists():                  # training size not run (e.g. --regime full)
            return "–"
        z = np.load(path, allow_pickle=True)
        t, e, r = z["times"].astype(float), z["events"].astype(int), z["risk"].astype(float)
        k = len(t) // n_test
        T, E, P = t.reshape(k, n_test), e.reshape(k, n_test), r.reshape(k, n_test)
        t, e = T[0], E[0]
        v, lo, hi, _ = bootstrap_ci(lambda i: float(np.mean([cindex(t[i], e[i], p[i]) for p in P])), t, e, B=B)
        return ci(v, lo, hi) + ("" if k in (1, 5) else f" †{k}")

    def preds(path):
        d = np.load(path, allow_pickle=True)
        return pd.DataFrame({"t": d["times"], "e": d["events"].astype(int), "r": d["risk"]}, index=d["keys"])

    def paired(pa, pb):
        a = preds(pa); b = preds(pb).loc[a.index]
        T, E = a.t.values, a.e.values
        return paired_bootstrap(lambda i: cindex(T[i], E[i], a.r.values[i]), lambda i: cindex(T[i], E[i], b.r.values[i]), T, E, B=B)

    res = {"S5": [], "S6": [], "S7": []}
    for co in ("Radcure", "Maastro"):
        full = pd.read_csv(_one(root, f"results_original_n8_*/_cells/rows_full_{co}_os_primary.csv"))
        st = full[full.arm == "ctfm_st"].iloc[0]
        n_test, n_ev, n_full = int(st.n_test), int(st.n_events), int(st.n_train)
        pdir = _one(root, "results_original_n8_*/ctfm_st/predictions")
        cells = [cell(pdir / f"{co}_os_primary_fewshot_n{n}.npz", n_test) for n in (10, 24, 50, 100)]
        res["S5"].append(["OS", co, LAB] + cells + [f"{cell(pdir / f'{co}_os_primary_fullshot.npz', n_test)} [{n_full}]"])
        d, lo, hi, p = paired(pdir / f"{co}_os_primary_fullshot.npz",
                              _one(main, f"results_original_n8_*/ctfm_st/predictions/{co}_os_primary_fullshot.npz"))
        c = pd.read_csv(_one(root, f"results_original_n8_*/_cells/comps_full_{co}_os_primary.csv"))
        k = c[(c.metric == "cindex") & (c.model_a == "combined_A") & (c.model_b == "clinical_only")].iloc[0]
        nt = f"{n_test} ({n_ev})"
        res["S6"] += [["Internal", "OS", co, nt, f"{LAB} − CT-FM-ST", dci(d, lo, hi), pf(p), ""],
                      ["Internal", "OS", co, nt, f"{LABC} − Clinical only", dci(k.delta, k.lo, k.hi), pf(k.p), ""]]
    er = pd.read_csv(_one(root, "results_external_original_n8_*/external_results_*primary.csv"))
    ec = pd.read_csv(_one(root, "results_external_original_n8_*/external_comparisons_*primary.csv"))
    er, ec = er[er.endpoint_code == "os"], ec[(ec.endpoint_code == "os") & (ec.metric == "cindex")]
    s, cb = er[er.arm == "ctfm_st"].iloc[0], er[er.arm == "combined_A"].iloc[0]
    nt = f"{int(s.n_target)} ({int(s.n_events)})"
    d, lo, hi, p = paired(_one(root, "results_external_original_n8_*/predictions/Radcure_to_Maastro_os_ctfm_st_refit_train_primary.npz"),
                          _one(main, "results_external_original_n8_*/predictions/Radcure_to_Maastro_os_ctfm_st_refit_train_primary.npz"))
    k1 = ec[(ec.model_a == "ctfm_st") & (ec.model_b == "clinical_only")].iloc[0]
    k2 = ec[(ec.model_a == "combined_A") & (ec.model_b == "clinical_only")].iloc[0]
    pp = lambda q: "p < 0.001" if q < 0.001 else f"p = {q:.3f}"
    res["S6"] += [["External", "OS", "Radcure → Maastro", nt, f"{LAB} − CT-FM-ST", dci(d, lo, hi), pf(p), ""],
                  ["External", "OS", "Radcure → Maastro", nt, f"{LABC} − Clinical only", dci(k2.delta, k2.lo, k2.hi), pf(k2.p), ""]]
    res["S7"] += [["OS", LAB, nt, ci(s.cindex, s.cindex_lo, s.cindex_hi), ci(s.ibs, s.ibs_lo, s.ibs_hi),
                   f"{dci(k1.delta, k1.lo, k1.hi)}, {pp(k1.p)}"],
                  ["OS", LABC, nt, ci(cb.cindex, cb.cindex_lo, cb.cindex_hi), ci(cb.ibs, cb.ibs_lo, cb.ibs_hi),
                   f"{dci(k2.delta, k2.lo, k2.hi)}, {pp(k2.p)}"]]
    out.mkdir(parents=True, exist_ok=True)
    (out / "spectre_cox_rows.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    head = {"S5": ["Endpoint", "Cohort", "Model", "10", "24", "50", "100", "Full-shot [N]"],
            "S6": ["Setting", "Endpoint", "Cohort", "N test (events)", "Comparison", "ΔC-index (95% CI)", "p", "LRT"],
            "S7": ["Endpoint", "Model", "N (events)", "C-index (95% CI)", "IBS (95% CI)", "ΔC-index vs clinical (95% CI), p"]}
    md = ["# SPECTRE + Cox head: rows of Supplementary Tables S5-S7 (OS)", ""]
    for t in ("S5", "S6", "S7"):
        md += [f"## Table {t}", "", "| " + " | ".join(head[t]) + " |", "|" + "---|" * len(head[t])]
        md += ["| " + " | ".join(r) + " |" for r in res[t]] + [""]
    (out / "spectre_cox_rows.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("a1"); p.add_argument("--cohort", choices=["Radcure", "Maastro"], required=True)
    p.add_argument("--regime", choices=["all", "full", "few"], default="all", help="all = few + full shot (published)")
    p.add_argument("--root", required=True); p.add_argument("--run", required=True)
    p = sp.add_parser("external"); p.add_argument("--root", required=True); p.add_argument("--run", required=True)
    p = sp.add_parser("rows"); p.add_argument("--root", required=True); p.add_argument("--main", required=True)
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "a1":
        run_script(a.root, a.run, ["run_a1.py", "--cohort", a.cohort, "--endpoint", "os", "--regime", a.regime])
    elif a.cmd == "external":
        run_script(a.root, a.run, ["run_external.py", "--spec", "primary", "--endpoint", "os"])
    else:
        rows(Path(a.root).resolve(), Path(a.main).resolve(), Path(a.out).resolve())
