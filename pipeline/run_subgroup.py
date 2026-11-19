"""Subgroup analysis on the held-out test sets -- Figure 4 / Table S5, with 95% CIs.

Replaces working_space/infer_subgroups_single_institution_fullshot.py, which re-extracts
features at inference and scores them through a saved bundle. That path has a confirmed
defect -- the OS bundles carry an identity scaler (mean 0, sd 1) although their Cox
coefficients were fitted on standardized features -- and its subgroup "All" C-index sits
0.09-0.11 below the full-shot number for EVERY endpoint, a gap that was never explained.
The test split is not the cause: the two scripts select the same 202 Radcure patients, in
the same order, for all four endpoints (checked).

This script re-uses the predictions run_a1.py already saved for those same test patients,
so nothing is re-extracted and no bundle is involved. The consequence worth stating: the
"All" row here EQUALS the full-shot result by construction, and the script checks that
against the run's own table. Any future gap is then a real finding, not a pipeline bug.

Models fitted with 5 seeds at full shot (CT-FM-LP, SegRes-E2E from run_baselines.py) get
the mean C-index over seeds, with the CI from bootstrapping the subgroup's patients and
recomputing that mean in every resample -- the same method as make_table_s4.py.

Subgroups follow the submitted Figure 4: whole test set, sex, HPV, T stage and N stage.
Each cell is a C-index with a 95% CI (2,000 event-stratified bootstrap resamples of the
patients in that subgroup), which is what R3 asked for; a subgroup with fewer than
MIN_EVENTS events gets a point estimate without a CI, and one with fewer than 2 events is
dropped. Inside a subgroup its defining variable is constant and can no longer rank
patients, so a model that leans on it is expected to fall back toward 0.5 there.

    python run_subgroup.py --runs results_corrected_hu_n8_<id> [more runs ...]
    python run_subgroup.py --runs ... --spec ajcc
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from bootstrap_ci import bootstrap_ci, cindex
from cohort import CLINICAL_COLS, ENDPOINT_PAPER_NAME, ENDPOINTS, build_cohort
from ipcw_brier import ibs_from_probs

COHORTS = ["Radcure", "Maastro"]
B_BOOT = 2000
MIN_EVENTS = 5            # below this a CI is not meaningful
MIN_EVENTS_ANY = 2        # below this not even a point estimate


def subgroup_masks(clin: pd.DataFrame) -> dict:
    """{label: boolean mask}, in the order the submitted Figure 4 uses."""
    out = {"All": np.ones(len(clin), bool)}
    for v in (0, 1):
        out[f"Sex={v}"] = (clin["sex"].astype(int) == v).values
    for v in (0, 1):
        out[f"HPV={v}"] = (clin["HPV"].astype(int) == v).values
    for v in (1, 2, 3, 4):
        out[f"T{v}"] = (clin["T_Stage"].astype(int) == v).values
    for v in (0, 1, 2, 3):
        out[f"N{v}"] = (clin["N_Stage"].astype(int) == v).values
    return out


def _arms(run: Path, imaging: str) -> dict:
    out = {}
    for d in sorted(p for p in run.iterdir() if (p / "predictions").is_dir()):
        out[f"combined_A:{imaging}" if d.name == "combined_A" else d.name] = d / "predictions"
    return out


LABEL = {"clinical_only": "Clinical only", "ctfm_st": "CT-FM-ST", "ctfm_lp": "CT-FM-LP",
         "segres_e2e": "SegRes-E2E", "segres_ft": "SegRes-FT", "spectre": "SPECTRE",
         "combined_A:ctfm_st": "Clinical + CT-FM", "combined_A:segres_ft": "Clinical + SegRes-FT",
         "combined_A:segres_e2e": "Clinical + SegRes-E2E"}
ORDER = list(LABEL)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--spec", choices=["primary", "ajcc"], default="primary")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    runs = [Path(r).resolve() for r in a.runs]
    infos = {r: (json.loads((r / "run_info.json").read_text()) if (r / "run_info.json").exists() else {})
             for r in runs}
    for f in ("data_version", "feature_version", "nstage"):
        vals = {i.get(f) for i in infos.values() if i}
        if len(vals) > 1:
            raise SystemExit(f"runs differ in {f}: {vals} -- they cannot share one table")
    # same version name is not enough: "corrected" was rebuilt in place (RFS fix), so
    # compare the md5 of the data files each run actually read
    md5s = {}
    for r in runs:
        info = json.loads((r / "run_info.json").read_text()) if (r / "run_info.json").exists() else {}
        md5s[r.name] = tuple(sorted((c, v.get("md5")) for c, v in info.get("data_files", {}).items()))
    if len({m for m in md5s.values() if m}) > 1:
        raise SystemExit("these runs read different data files (md5 differ), so their test labels "
                         "differ and they cannot share one table:\n   " +
                         "\n   ".join(f"{k}: {dict(v)}" for k, v in md5s.items()))
    sources = {}
    for r in runs:
        for arm, pred in _arms(r, infos[r].get("imaging", "ctfm_st")).items():
            sources.setdefault(arm, pred)
    # SPECTRE is not in the submitted Figure 4 (and was OS-only in the paper)
    arms = [x for x in ORDER if x in sources and x != "spectre"] + sorted(set(sources) - set(ORDER))
    print(f"models: {', '.join(LABEL.get(x, x) for x in arms)}")

    rows, checked = [], []
    for co in COHORTS:
        for ep in ENDPOINTS:
            # clinical covariates of the test patients, joined by patient key
            clin_all = build_cohort(co, ep).set_index("key")
            for arm in arms:
                f = sources[arm] / f"{co}_{ep}_{a.spec}_fullshot.npz"
                if not f.exists():        # imaging-only baselines use no clinical variable
                    f = sources[arm] / f"{co}_{ep}_primary_fullshot.npz"
                if not f.exists():
                    continue
                z = np.load(f, allow_pickle=True)
                keys = z["keys"]
                n0 = len(keys)
                # run_baselines.py stacks 5 seeds' predictions for the same patients; a
                # single fit is the one-seed case
                k = int(z["n_seeds"]) if "n_seeds" in z.files else len(z["times"]) // n0
                t, e = z["times"].astype(float)[:n0], z["events"].astype(int)[:n0]
                R = z["risk"].astype(float).reshape(k, n0)
                S = z["surv"].reshape(k, n0, -1) if z["surv"].size else None
                et = z["eval_times"] if "eval_times" in z.files else None
                clin = clin_all.loc[keys, CLINICAL_COLS].reset_index(drop=True)
                for name, m in subgroup_masks(clin).items():
                    n, ne = int(m.sum()), int(e[m].sum())
                    if ne < MIN_EVENTS_ANY:
                        continue
                    tm, em = t[m], e[m]
                    # mean over seeds, recomputed inside every bootstrap resample of patients
                    c_of = lambda i: float(np.mean([cindex(tm[i], em[i], r[m][i]) for r in R]))
                    rec = {"cohort": co, "endpoint": ENDPOINT_PAPER_NAME[ep],
                           "model": LABEL.get(arm, arm), "subgroup": name, "n": n, "events": ne,
                           "n_seeds": k, "cindex": np.nan, "lo": np.nan, "hi": np.nan, "ibs": np.nan}
                    if ne >= MIN_EVENTS:
                        p, lo, hi, _ = bootstrap_ci(c_of, tm, em, B=B_BOOT)
                        rec.update(cindex=p, lo=lo, hi=hi)
                        if S is not None and np.isfinite(S[:, m]).all():
                            rec["ibs"] = float(np.mean([ibs_from_probs(sv[m], tm, em, et) for sv in S]))
                    else:
                        rec["cindex"] = c_of(np.arange(n))
                    rows.append(rec)
                    if name == "All":
                        checked.append((co, ENDPOINT_PAPER_NAME[ep], LABEL.get(arm, arm), rec["cindex"]))
    df = pd.DataFrame(rows)

    # ---- the check the old pipeline fails: "All" must equal the full-shot result
    ok, mismatched = 0, []
    for r in runs:
        for f in r.glob(f"_cells/rows_full_*_{a.spec}.csv"):
            t = pd.read_csv(f)
            for _, x in t[t.regime == "full_shot"].iterrows():
                lab = LABEL.get(f"combined_A:{infos[r].get('imaging', 'ctfm_st')}"
                                if x.arm == "combined_A" else x.arm, x.arm)
                hit = [c for c in checked if c[:3] == (x.cohort, x.endpoint, lab)]
                if not hit or not np.isfinite(x.cindex):
                    continue
                if abs(hit[0][3] - x.cindex) <= 1e-9:
                    ok += 1
                else:
                    mismatched.append(f"{x.cohort}/{x.endpoint}/{lab}: "
                                      f"{hit[0][3]:.4f} vs {x.cindex:.4f}")
    if ok or mismatched:
        print(f"check: subgroup 'All' equals the full-shot C-index for {ok} cells"
              + (f"  -- {len(mismatched)} MISMATCH: {mismatched[:3]}" if mismatched else "  (0 mismatches)"))
    else:
        print("check: no run_a1.py full-shot results in these runs to compare 'All' against")

    out = Path(a.out) if a.out else runs[0] / ("_report" if a.spec == "primary" else f"_report_{a.spec}")
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "table_s5_subgroups_long.csv", index=False)

    # ---- wide table, one row per cohort x endpoint x model
    cell = lambda r: ("—" if not np.isfinite(r.cindex) else
                      f"{r.cindex:.3f} ({r.lo:.3f}–{r.hi:.3f})" if np.isfinite(r.lo)
                      else f"{r.cindex:.3f}*")
    wide = []
    order = list(subgroup_masks(build_cohort("Radcure", "os").head(1)))
    for (co, ep, mod), g in df.groupby(["cohort", "endpoint", "model"], sort=False):
        row = {"Cohort": co, "Endpoint": ep, "Model": mod}
        for s in order:
            h = g[g.subgroup == s]
            row[f"{s} (n)"] = "—" if h.empty else f"{cell(h.iloc[0])} [{int(h.iloc[0].n)}]"
        wide.append(row)
    W = pd.DataFrame(wide)
    W.to_csv(out / "table_s5_subgroups.csv", index=False)
    cols = list(W.columns)
    w = [max(len(str(c)), *(len(str(v)) for v in W[c])) for c in cols]
    note = (f"C-index (95% CI) [n] on the held-out test sets, by subgroup. CIs: {B_BOOT:,} "
            f"event-stratified bootstrap resamples within the subgroup; * = fewer than "
            f"{MIN_EVENTS} events, point estimate only; subgroups with fewer than {MIN_EVENTS_ANY} "
            f"events are omitted. Within a subgroup its defining variable is constant and cannot "
            f"rank patients. 'All' reproduces the full-shot result exactly (same patients, same "
            f"saved predictions).")
    md = ["## Table S5 (revised). Subgroup discrimination, with 95% CI", "", note, "",
          "| " + " | ".join(str(c).ljust(w[k]) for k, c in enumerate(cols)) + " |",
          "|" + "|".join("-" * (x + 2) for x in w) + "|"]
    md += ["| " + " | ".join(str(r[c]).ljust(w[k]) for k, c in enumerate(cols)) + " |"
           for _, r in W.iterrows()]
    (out / "table_s5_subgroups.md").write_text("\n".join(md) + "\n")
    print(f"\nwrote {out / 'table_s5_subgroups.md'}\n      {out / 'table_s5_subgroups.csv'}"
          f"\n      {out / 'table_s5_subgroups_long.csv'}")

    import make_figure_s5
    make_figure_s5.main(str(out / "table_s5_subgroups_long.csv"), str(out / "figures"))


if __name__ == "__main__":
    main()
