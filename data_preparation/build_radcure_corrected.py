"""Build a corrected Radcure analysis file with PURE locoregional / distant endpoints.

Input (not modified):   data.radcure_csv            (config.yaml; the submitted analysis file)
Output:                 data.radcure_csv_corrected  (config.yaml)
                        <same folder>/<its name without "_main">_CHANGES.csv   (every changed cell)

Why: in the original file lr_event / dr_event are COMPOSITE -- death from any cause
is counted as an event (414 / 420). After the Maastro correction, Maastro uses PURE
endpoints (failure only). Both cohorts must use the same definition before results
can be compared or one cohort used to validate the other.

There is no pure Radcure export, so the pure events are derived from the composite
ones with one rule: a composite event is dropped only if it IS the death, i.e. it
falls on the same day as a recorded death and nothing happened earlier:

    pure = composite_event == 1  AND  NOT (event_time == os_time AND os_event == 1)

Validated before use: applied to the original Maastro file, this rule reproduces
the second, pure-event Maastro export (data.maastro_pure_events_csv) for 173/173 patients
on both LR and DM.

Time columns are unchanged: dropped events already sit at os_time, which is exactly
where a death-censored pure endpoint censors them.

Derived 2yr/3yr LR/DM columns are rebuilt from the pure events with the original files'
own rule (derived_cols.horizon_cells): a patient whose follow-up ends before the horizon
without a failure -- including one who died first -- is "censored" in both the event and
time cell. OS derived columns are untouched (their base columns do not change).

RFS TIME (efs_time) is also corrected. RFS ends at the FIRST of recurrence (LR or DM) or
death. In the original file efs_time is instead the death time for every patient who
recurred and later died (184), and the later of two recurrences for 2 patients with both
LR and DM -- 186 patients in all, median 190 days too late. Maastro has it right.
efs_event is already correct and is not changed. The rule:

    efs_time = min(lr_time if lr_event, dr_time if dr_event, os_time if os_event)
               for every efs_event == 1 patient; censored patients unchanged

Every change moves the time EARLIER, onto a recorded LR or DM date (checked below). The
2yr/3yr RFS columns of the changed patients are rebuilt with the same horizon rule.

Known pre-existing data issues, left untouched and reported:
  * one patient has a dr/efs event recorded after os_time (last contact, alive).
  * two patients have an event exactly on a 2yr/3yr horizon day with a derived time that
    does not match the base time.
(The revision code named these patients; their identifiers are not published here, so the
check below counts the first kind instead of naming it.)

    python build_radcure_corrected.py
"""
from __future__ import annotations

import csv
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from derived_cols import HORIZONS, horizon_cells

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
from ctfm_config import CFG  # noqa: E402

SRC = CFG.path("data.radcure_csv", required=True)
OUT = CFG.path("data.radcure_csv_corrected", required=True)
LOG = OUT.with_name(OUT.stem.removesuffix("_main") + "_CHANGES.csv")
KEY = "img_nn_id"
ENDPOINTS = ["lr", "dr"]
N_KNOWN_ANOMALIES = 1       # patients with an event recorded after last contact (see docstring)


def pure_events(d: pd.DataFrame, e: str) -> np.ndarray:
    death_only = (d[f"{e}_time"] == d["os_time"]) & (d["os_event"] == 1)
    return ((d[f"{e}_event"] == 1) & ~death_only).astype(int).values


def first_event_times(d: pd.DataFrame, pure: dict) -> np.ndarray:
    """RFS time = the first of a pure LR, a pure DM or death (inf if none of them)."""
    inf = np.inf
    return np.minimum.reduce([
        np.where(pure["lr"] == 1, d["lr_time"], inf),
        np.where(pure["dr"] == 1, d["dr_time"], inf),
        np.where(d["os_event"] == 1, d["os_time"], inf),
    ])


def main():
    with open(SRC, newline="") as fh:
        rows = list(csv.reader(fh))
    header, rows = rows[0], rows[1:]
    col = {c: i for i, c in enumerate(header)}
    src = pd.read_csv(SRC)
    if (src[KEY].astype(str).values != np.array([r[col[KEY]] for r in rows])).any():
        raise SystemExit("row order mismatch between csv and pandas read")

    changes = []

    def put(r, c, after):
        before = rows[r][col[c]]
        try:                                    # same number, different text -> no change
            if np.isclose(float(before), float(after)):
                return
        except ValueError:
            pass
        if before != after:
            changes.append({KEY: rows[r][col[KEY]], "column": c,
                            "old_value": before, "new_value": after})
            rows[r][col[c]] = after

    pure = {e: pure_events(src, e) for e in ENDPOINTS}
    for r in range(len(rows)):
        for e in ENDPOINTS:
            put(r, f"{e}_event", str(int(pure[e][r])))
            t_text = rows[r][col[f"{e}_time"]]
            for y, H in HORIZONS.items():           # the original files' own rule
                ev_cell, t_cell = horizon_cells(t_text, int(pure[e][r]), H)
                put(r, f"{y}_{e}_event", ev_cell)
                put(r, f"{y}_{e}_time", t_cell)

    # RFS time = first of recurrence or death (see docstring); event flags untouched
    first = first_event_times(src, pure)
    rfs_fixed = []
    for r in range(len(rows)):
        if src["efs_event"].iloc[r] != 1 or not np.isfinite(first[r]):
            continue
        if first[r] < src["efs_time"].iloc[r] - 1e-9:
            t_new = f"{first[r]:g}"
            put(r, "efs_time", t_new)
            rfs_fixed.append(r)
            for y, H in HORIZONS.items():
                ev_cell, t_cell = horizon_cells(t_new, 1, H)
                put(r, f"{y}_efs_event", ev_cell)
                put(r, f"{y}_efs_time", t_cell)

    # keep the original line endings / trailing-newline state
    raw = SRC.read_bytes()
    term = "\r\n" if b"\r\n" in raw else "\n"
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator=term)
    w.writerow(header); w.writerows(rows)
    text = buf.getvalue()
    if not raw.endswith(b"\n"):
        text = text[: -len(term)]
    with open(OUT, "w", newline="") as fh:
        fh.write(text)
    pd.DataFrame(changes).to_csv(LOG, index=False)

    # ---------------------------------------------------------------- validation
    fix = pd.read_csv(OUT)
    death_only_left = sum(int(((fix[f"{e}_event"] == 1) & (fix[f"{e}_time"] == fix.os_time)
                               & (fix.os_event == 1)).sum()) for e in ENDPOINTS)
    after_os = set()
    for e in ENDPOINTS + ["efs"]:
        after_os |= set(fix.loc[(fix[f"{e}_event"] == 1) & (fix[f"{e}_time"] > fix.os_time), KEY])
    changed_cols = {c["column"] for c in changes}
    allowed = ({f"{e}_event" for e in ENDPOINTS}
               | {f"{y}_{e}_{k}" for y in HORIZONS for e in ENDPOINTS for k in ("event", "time")}
               | {"efs_time"} | {f"{y}_efs_{k}" for y in HORIZONS for k in ("event", "time")})
    raw_fix = pd.read_csv(OUT, dtype=str, keep_default_na=False)
    derived_ok = all(
        horizon_cells(raw_fix[f"{e}_time"].iloc[i], int(raw_fix[f"{e}_event"].iloc[i]), H)[0]
        == raw_fix[f"{y}_{e}_event"].iloc[i].strip()
        for e in ENDPOINTS for y, H in HORIZONS.items() for i in range(len(raw_fix)))
    # RFS: which rows changed, and is the new time a recorded LR / DM date?
    ev = fix["efs_event"] == 1
    moved = fix["efs_time"].values < src["efs_time"].values - 1e-9
    on_failure_date = [
        (fix["lr_event"].iloc[i] == 1 and np.isclose(fix["efs_time"].iloc[i], fix["lr_time"].iloc[i]))
        or (fix["dr_event"].iloc[i] == 1 and np.isclose(fix["efs_time"].iloc[i], fix["dr_time"].iloc[i]))
        for i in rfs_fixed]
    rfs_derived_ok = all(
        horizon_cells(raw_fix["efs_time"].iloc[i], 1, H)[0] == raw_fix[f"{y}_efs_event"].iloc[i].strip()
        for i in rfs_fixed for y, H in HORIZONS.items())
    untouched = [c for c in src.columns if c not in allowed]
    checks = {
        "same shape": fix.shape == src.shape,
        "same column order": list(fix.columns) == list(src.columns),
        "same patient order": (fix[KEY].values == src[KEY].values).all(),
        "only intended columns changed": changed_cols <= allowed,
        "every base LR/DM change is 1 -> 0": all(
            c["old_value"] == "1" and c["new_value"] == "0"
            for c in changes if c["column"] in {f"{e}_event" for e in ENDPOINTS}),
        "derived 2yr/3yr LR/DM columns follow the rule": derived_ok,
        "no death-only events remain": death_only_left == 0,
        "RFS event flags unchanged": (fix["efs_event"].values == src["efs_event"].values).all(),
        "RFS times only moved earlier, only for RFS events":
            (fix["efs_time"].values <= src["efs_time"].values + 1e-9).all()
            and set(np.flatnonzero(moved)) == set(rfs_fixed) and ev.values[moved].all(),
        "every moved RFS time is a recorded LR or DM date": all(on_failure_date),
        "RFS time = first of recurrence or death, every RFS event":
            np.allclose(fix.loc[ev, "efs_time"].values, first[ev.values]),
        "derived 2yr/3yr RFS columns follow the rule (moved rows)": rfs_derived_ok,
        "events after os_time only the known anomaly": len(after_os) <= N_KNOWN_ANOMALIES,
        "all other columns identical": all(
            src[c].astype(str).equals(fix[c].astype(str)) for c in untouched),
        "line endings preserved": (b"\r\n" in OUT.read_bytes()) == (b"\r\n" in raw),
    }
    print(f"wrote {OUT.name}   ({fix.shape[0]} rows x {fix.shape[1]} cols)")
    print(f"wrote {LOG.name}   ({len(changes)} changed cells)\n")
    print("changed cells per column:")
    for c in sorted(changed_cols):
        print(f"   {c:<16} {sum(x['column'] == c for x in changes)}")
    print("\nvalidation:")
    for k, v in checks.items():
        print(f"   {'PASS' if v else 'FAIL'}  {k}")
    if not all(checks.values()):
        raise SystemExit("\nVALIDATION FAILED -- do not use the output")
    print("\nevent counts (original -> corrected):")
    for e in ["os", "efs", "lr", "dr"]:
        a, b = int(src[f"{e}_event"].sum()), int(fix[f"{e}_event"].sum())
        print(f"   {e:<4} {a:>4} -> {b:>4}  ({100 * b / len(fix):.1f}%)")
    shift = (src["efs_time"] - fix["efs_time"]).values[moved]
    print(f"\nRFS time moved to the first recurrence for {len(rfs_fixed)} patients "
          f"(median {np.median(shift):.0f} days earlier, range {shift.min():.0f}-{shift.max():.0f})")
    print(f"\nknown anomaly left untouched: {len(after_os)} patient(s) "
          f"(event recorded after last contact -- confirm with data owner)")


if __name__ == "__main__":
    main()
