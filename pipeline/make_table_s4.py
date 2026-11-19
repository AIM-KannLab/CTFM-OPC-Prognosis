"""Table S4 of the manuscript, rebuilt with 95% confidence intervals.

Rows: endpoint x cohort x model. Columns: training-set size -- 10, 24, 50, 100 patients
and full shot [N], as in the submitted Table S4. Each cell: C-index (95% CI), where the
point estimate is the mean over seeds, the number the submitted table reported.

The CI belongs to that mean. It bootstraps the held-out TEST PATIENTS: every resample is
applied to all seeds' predictions at once, each seed's C-index is recomputed on it, and
the seeds are averaged. Seed-to-seed spread stays visible as the SD (in the CSV). A single
deterministic fit -- CT-FM-ST and the clinical models at full shot -- is the one-seed case,
and then reproduces make_report's full-shot CI exactly.

Do NOT use the "seed-pooled" CI of make_report's few-shot table instead: it stacks the
seeds' scores for the same patients as if they were different patients, so it is too
narrow -- by a median 1.5x (up to ~2x) on the CT-FM run of 2026-10-02.

Nothing is refitted: the per-seed test predictions run_a1.py saved are read back. Several
run folders can be combined (e.g. the CT-FM run and the SegRes-FT run); each model comes
from the first folder that has it. The folders must share data version, input and N-stage
setting (checked), so every row is scored on the same test patients.

    python make_table_s4.py --runs results_corrected_hu_n8_<id> results_corrected_hu_n8_segresft_<id>
    python make_table_s4.py --runs ... --sizes all          # every few-shot size, not just 10-100
    python make_table_s4.py --runs ... --metric ibs         # integrated Brier score instead
"""
from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from bootstrap_ci import bootstrap_ci, cindex
from cohort import ENDPOINT_PAPER_NAME, ENDPOINTS
from ipcw_brier import ibs_from_probs

PAPER_SIZES = [10, 24, 50, 100]
COHORTS = ["Radcure", "Maastro"]
B_BOOT = 2000
# Row order: the reviewers' clinical baseline first, then the imaging models in the order
# of the submitted table, then the combined models.
ORDER = ["clinical_only", "ctfm_st", "ctfm_lp", "segres_e2e", "segres_ft", "spectre",
         "combined_A:ctfm_st", "combined_A:segres_ft", "combined_A:segres_e2e"]
LABEL = {"clinical_only": "Clinical only", "ctfm_st": "CT-FM-ST", "ctfm_lp": "CT-FM-LP",
         "segres_e2e": "SegRes-E2E", "segres_ft": "SegRes-FT", "spectre": "SPECTRE",
         "combined_A:ctfm_st": "Clinical + CT-FM", "combined_A:segres_ft": "Clinical + SegRes-FT",
         "combined_A:segres_e2e": "Clinical + SegRes-E2E"}
PAPER_MODELS = ["ctfm_st", "ctfm_lp", "segres_e2e", "spectre"]


def _run_info(run: Path) -> dict:
    p = run / "run_info.json"
    info = json.loads(p.read_text()) if p.exists() else {}
    return {"imaging": info.get("imaging", "ctfm_st"), "data": info.get("data_version", "?"),
            "features": info.get("feature_version", "published"), "nstage": info.get("nstage", "raw")}


def same_data_or_exit(runs) -> None:
    """Refuse runs that read different data files. The version name is not enough:
    "corrected" was rebuilt in place (the RFS fix), so compare the md5s each run recorded."""
    md5s = {}
    for r in runs:
        info = json.loads((r / "run_info.json").read_text()) if (r / "run_info.json").exists() else {}
        md5s[r.name] = tuple(sorted((c, v.get("md5")) for c, v in info.get("data_files", {}).items()))
    if len({m for m in md5s.values() if m}) > 1:
        raise SystemExit("these runs read different data files (md5 differ), so their test labels "
                         "differ and they cannot share one table:\n   " +
                         "\n   ".join(f"{k}: {dict(v)}" for k, v in md5s.items()))


def _arms(run: Path, imaging: str) -> dict:
    """{row id: prediction folder}. combined_A is renamed after the run's imaging model,
    because 'Clinical + CT-FM' and 'Clinical + SegRes-FT' share that folder name."""
    out = {}
    for d in sorted(p for p in run.iterdir() if (p / "predictions").is_dir()):
        out[f"combined_A:{imaging}" if d.name == "combined_A" else d.name] = d / "predictions"
    return out


def _load(path: Path, n_test: int | None, metric: str):
    """Times, events and one row of predictions per seed, from a saved .npz.

    Files written by run_baselines.py state their seed count (n_seeds); for run_a1.py's
    files it is the row count over the test-set size."""
    z = np.load(path, allow_pickle=True)
    t, e = z["times"].astype(float), z["events"].astype(int)
    if "n_seeds" in z.files:
        n_test = len(t) // int(z["n_seeds"])
    if not n_test or len(t) % n_test:
        raise ValueError(f"{path.name}: {len(t)} rows is not a whole number of {n_test}-patient seeds")
    k = len(t) // n_test
    T, E = t.reshape(k, n_test), e.reshape(k, n_test)
    if not (np.all(T == T[0]) and np.all(E == E[0])):
        raise ValueError(f"{path.name}: the seeds were not scored on the same test patients")
    if metric == "cindex":
        P = z["risk"].astype(float).reshape(k, n_test)
    else:
        if z["surv"].size == 0:
            return None
        P = z["surv"].astype(float).reshape(k, n_test, -1)
    return T[0], E[0], P, z["eval_times"] if "eval_times" in z.files else None


def _cell(job):
    """One table cell: mean over seeds, its bootstrap 95% CI, and the SD over seeds."""
    key, path, n_test, metric = job
    got = _load(path, n_test, metric)
    if got is None:
        return key, None
    t, e, P, et = got
    if metric == "cindex":
        one = lambda i, p: cindex(t[i], e[i], p[i])
    else:
        one = lambda i, p: ibs_from_probs(p[i], t[i], e[i], et)
    per_seed = [one(np.arange(len(t)), p) for p in P]
    point, lo, hi, nv = bootstrap_ci(lambda i: float(np.mean([one(i, p) for p in P])),
                                     t, e, B=B_BOOT)
    return key, {"value": point, "lo": lo, "hi": hi, "sd": float(np.std(per_seed, ddof=1))
                 if len(per_seed) > 1 else 0.0, "n_seeds": len(P), "n_test": len(t),
                 "n_events": int(e.sum()), "valid_resamples": nv}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", nargs="+", required=True, help="run folders (results_...)")
    ap.add_argument("--spec", choices=["primary", "ajcc"], default="primary")
    ap.add_argument("--metric", choices=["cindex", "ibs"], default="cindex")
    ap.add_argument("--sizes", choices=["paper", "all"], default="paper",
                    help="paper = 10, 24, 50, 100 + full shot (as Table S4); all = every size run")
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--no-figures", action="store_true", help="table only (figures: make_figures_s4.py)")
    ap.add_argument("--out", default=None, help="output folder (default: <first run>/_report[_spec])")
    a = ap.parse_args()

    runs = [Path(r).resolve() for r in a.runs]
    infos = {r: _run_info(r) for r in runs}
    for field in ("data", "features", "nstage"):
        vals = {i[field] for i in infos.values()}
        if len(vals) > 1:
            raise SystemExit(f"runs differ in {field}: {vals} -- their test patients or labels "
                             f"may differ, so they cannot share one table")
    same_data_or_exit(runs)
    sources = {}
    for r in runs:                                   # first folder that has a model wins
        for arm, pred in _arms(r, infos[r]["imaging"]).items():
            sources.setdefault(arm, pred)
    arms = [x for x in ORDER if x in sources] + sorted(set(sources) - set(ORDER))
    print(f"models: {', '.join(LABEL.get(x, x) for x in arms)}")
    missing = [LABEL[m] for m in PAPER_MODELS if m not in sources]
    if missing:
        print(f"   not in these runs yet (rows will be absent): {', '.join(missing)}")

    jobs, sizes_seen, n_full = [], {}, {}
    for co in COHORTS:
        for ep in ENDPOINTS:
            # test-set size, from a single-fit file; run_baselines.py files carry n_seeds instead
            ref = sources.get("clinical_only")
            ref = ref / f"{co}_{ep}_{a.spec}_fullshot.npz" if ref else None
            n_test = len(np.load(ref, allow_pickle=True)["times"]) if ref and ref.exists() else None
            for arm in arms:
                pred = sources[arm]
                # imaging-only baselines use no clinical variable: their files are "primary"
                spec = a.spec if any(pred.glob(f"{co}_{ep}_{a.spec}_*.npz")) else "primary"
                few = sorted(int(p.stem.rsplit("_n", 1)[1])
                             for p in pred.glob(f"{co}_{ep}_{spec}_fewshot_n*.npz"))
                sizes_seen.setdefault(co, set()).update(few)
                for s in few:            # every size: the curves need them; --sizes trims only the table
                    jobs.append(((co, ep, arm, s), pred / f"{co}_{ep}_{spec}_fewshot_n{s}.npz",
                                 n_test, a.metric))
                full = pred / f"{co}_{ep}_{spec}_fullshot.npz"
                if full.exists():
                    jobs.append(((co, ep, arm, "full"), full, n_test, a.metric))
                    z = np.load(full, allow_pickle=True)
                    if "n_train" in z.files:
                        n_full.setdefault(co, int(z["n_train"]))
    # full-shot training size per cohort, from the runs' own result rows (else from the files)
    for co in COHORTS:
        f = next((p for r in runs for p in r.glob(f"_cells/rows_full_{co}_*_{a.spec}.csv")), None)
        if f is not None:
            n_full[co] = int(pd.read_csv(f).n_train.iloc[0])
    print(f"{len(jobs)} cells, {B_BOOT:,} bootstrap resamples each, {a.procs} processes ...", flush=True)
    with Pool(a.procs) as pool:
        res = dict(r for r in pool.imap_unordered(_cell, jobs, chunksize=1) if r[1] is not None)

    # ------------------------------------------------------------------ tables
    def size_cols(co):
        few = sorted(sizes_seen.get(co, set()))
        return (few if a.sizes == "all" else [s for s in PAPER_SIZES if s in few]) + ["full"]
    # long format: every computed cell, every training size (the figures read this)
    rec = [{"endpoint": ENDPOINT_PAPER_NAME[ep], "cohort": co, "model": LABEL.get(arm, arm),
            "train_size": s, "n_train_full": n_full.get(co), **c}
           for (co, ep, arm, s), c in sorted(res.items(), key=lambda kv: str(kv[0]))]
    md_rows = []
    for ep in ENDPOINTS:
        for co in COHORTS:
            for arm in arms:
                cells = {s: res.get((co, ep, arm, s)) for s in size_cols(co)}
                if not any(cells.values()):
                    continue
                row = {"Endpoint": ENDPOINT_PAPER_NAME[ep], "Cohort": co, "Model": LABEL.get(arm, arm)}
                for s, c in cells.items():
                    col = "Full-shot [N]" if s == "full" else str(s)   # as in the submitted table
                    if c is None:
                        row[col] = "—"
                        continue
                    mark = "" if c["n_seeds"] in (1, 5) else f" †{c['n_seeds']}"
                    size = f" [{n_full.get(co, '?')}]" if s == "full" else ""
                    row[col] = f"{c['value']:.3f} ({c['lo']:.3f}–{c['hi']:.3f}){mark}{size}"
                md_rows.append(row)
    table = pd.DataFrame(md_rows).fillna("—")
    size_order = sorted({c for c in table.columns if c.isdigit()}, key=int)
    table = table[["Endpoint", "Cohort", "Model"] + size_order + ["Full-shot [N]"]]
    long = pd.DataFrame(rec)

    out = Path(a.out) if a.out else runs[0] / ("_report" if a.spec == "primary" else f"_report_{a.spec}")
    out.mkdir(parents=True, exist_ok=True)
    tag = f"table_s4_{a.metric}" + ("" if a.sizes == "paper" else "_all_sizes")
    long_path = out / f"table_s4_{a.metric}_long.csv"      # every training size, always
    long.to_csv(long_path, index=False)
    table.to_csv(out / f"{tag}.csv", index=False)
    i = next(iter(infos.values()))
    name = "C-index" if a.metric == "cindex" else "Integrated Brier score"
    note = (f"{name} on the held-out test sets (Radcure n=202, Maastro n=26), mean over 5 random "
            f"training subsets (seeds) with 95% CI. The CI bootstraps the test patients "
            f"({B_BOOT:,} event-stratified resamples, the same resample applied to every seed); "
            f"full-shot fits that are deterministic are a single fit. † = fewer than 5 seeds "
            f"could be fitted (number shown). Data: {i['data']}; input: {i['features']}; "
            f"N stage: {'AJCC, harmonised' if i['nstage'] == 'ajcc8' else 'as recorded per cohort'}. "
            f"Runs: {', '.join(r.name for r in runs)}.")
    cols = list(table.columns)
    w = [max(len(str(c)), *(len(str(v)) for v in table[c])) for c in cols]
    md = [f"## Table S4 (revised). {name} by training-set size, with 95% CI", "", note, "",
          "| " + " | ".join(str(c).ljust(w[k]) for k, c in enumerate(cols)) + " |",
          "|" + "|".join("-" * (x + 2) for x in w) + "|"]
    md += ["| " + " | ".join(str(r[c]).ljust(w[k]) for k, c in enumerate(cols)) + " |"
           for _, r in table.iterrows()]
    (out / f"{tag}.md").write_text("\n".join(md) + "\n")
    print(f"\nwrote {out / (tag + '.md')}\n      {out / (tag + '.csv')}\n      {long_path}")
    if a.metric == "cindex" and not a.no_figures:
        import make_figures_s4
        make_figures_s4.main(str(long_path), str(out / "figures"), n_full=n_full)


if __name__ == "__main__":
    main()
