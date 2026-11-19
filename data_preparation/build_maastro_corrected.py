"""Build a corrected Maastro analysis file from the two available exports.

Inputs (neither is modified; locations from config.yaml):
    OLD  data.maastro_csv                original analysis file (2025-06)
    NEW  data.maastro_pure_events_csv    second export with the pure LR / DM events (2026-10)

Output:
    data.maastro_csv_corrected
    <same folder>/<its name without "_main">_CHANGES.csv   (every changed cell)

What each export gets right, verified against the data:

    OLD  correct HPV coding     HPV=1 -> HR 0.32 for death (protective, matches Radcure
                                and the manuscript's 100 HPV-positive)
         correct time columns   every lr/dr/efs time <= os_time; censored rows carry os_time
         WRONG lr/dr events     composite: death counted as an event (89 / 86)

    NEW  correct lr/dr events   pure locoregional / distant failures only (31 / 27)
         WRONG HPV coding       0/1 inverted (HR 3.14)
         WRONG censoring times  139-144 rows have lr/dr time > os_time
         WRONG efs              37 RFS events dated after death

So the corrected file is the OLD file with exactly one substantive change -- the pure
lr_event / dr_event from NEW -- plus the 2yr/3yr lr/dr event and time columns rebuilt
from them with the original files' own rule (derived_cols.horizon_cells).

No time column changes. For the patients with an event, OLD and NEW times are
identical; for everyone else OLD already censors at os_time. That was verified,
and is re-asserted below.

The file is copied cell-for-cell as text and only the target cells are rewritten,
so every untouched value is byte-identical to the original.

    python build_maastro_corrected.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from derived_cols import HORIZONS, horizon_cells

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
from ctfm_config import CFG  # noqa: E402

OLD = CFG.path("data.maastro_csv", required=True)
NEW = CFG.path("data.maastro_pure_events_csv", required=True)
OUT = CFG.path("data.maastro_csv_corrected", required=True)
LOG = OUT.with_name(OUT.stem.removesuffix("_main") + "_CHANGES.csv")

KEY = "img_nn_id"
REPLACE = ["lr_event", "dr_event"]                       # taken from NEW


def read_text(path):
    """Read a CSV as raw strings so untouched cells round-trip byte-for-byte."""
    with open(path, newline="") as fh:
        rows = list(csv.reader(fh))
    return rows[0], rows[1:]


def main():
    header, rows = read_text(OLD)
    col = {c: i for i, c in enumerate(header)}
    old = pd.read_csv(OLD)
    new = pd.read_csv(NEW).set_index(KEY)

    # ---------------------------------------------------------------- alignment
    if len(rows) != len(new):
        raise SystemExit(f"row count differs: OLD {len(rows)} vs NEW {len(new)}")
    if set(old[KEY]) != set(new.index):
        raise SystemExit("patient sets differ between OLD and NEW")
    new = new.loc[old[KEY]]                              # NEW in OLD's row order

    # ---------------------------------------------------------------- pre-checks
    # The time columns must already be right in OLD, or this merge is not valid.
    for e in REPLACE:
        base = e.replace("_event", "")
        ev = new[e].values == 1
        t_old, t_new = old[f"{base}_time"].values, new[f"{base}_time"].values
        if not np.allclose(t_old[ev], t_new[ev]):
            raise SystemExit(f"{base}: event times differ between OLD and NEW -- stop")
        if not np.allclose(t_old[~ev], old["os_time"].values[~ev]):
            raise SystemExit(f"{base}: OLD censoring times are not os_time -- stop")
        if np.any(t_old[ev] > old["os_time"].values[ev]):
            raise SystemExit(f"{base}: an event time falls after os_time -- stop")

    # ---------------------------------------------------------------- patch cells
    changes = []

    def put(r, c, after):
        after = str(after)
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

    for r in range(len(rows)):
        for e in REPLACE:
            base = e.replace("_event", "")
            ev = int(new[e].iloc[r])
            put(r, e, ev)
            t_text = rows[r][col[f"{base}_time"]]
            for y, H in HORIZONS.items():               # the original files' own rule
                ev_cell, t_cell = horizon_cells(t_text, ev, H)
                put(r, f"{y}_{e}", ev_cell)
                put(r, f"{y}_{base}_time", t_cell)

    # Reproduce the original's line endings exactly. new_maastro_data.csv uses Windows
    # CRLF and has no trailing newline; writing Unix LF would make every line differ
    # in a diff even though no value changed.
    raw = OLD.read_bytes()
    term = "\r\n" if b"\r\n" in raw else "\n"
    trailing = raw.endswith(b"\n")
    import io
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator=term)
    w.writerow(header)
    w.writerows(rows)
    text = buf.getvalue()
    if not trailing:
        text = text[: -len(term)]
    with open(OUT, "w", newline="") as fh:
        fh.write(text)
    pd.DataFrame(changes).to_csv(LOG, index=False)

    # ---------------------------------------------------------------- validation
    from lifelines import CoxPHFitter
    fix = pd.read_csv(OUT)
    checks = {
        "same shape as OLD": fix.shape == old.shape,
        "same column order": list(fix.columns) == list(old.columns),
        "same patient order": (fix[KEY].values == old[KEY].values).all(),
        "lr_event == 31": int(fix.lr_event.sum()) == 31,
        "dr_event == 27": int(fix.dr_event.sum()) == 27,
        "lr/dr match NEW exactly": all((fix[e].values == new[e].values).all() for e in REPLACE),
        "no time > os_time": all((fix[f"{e}_time"] <= fix.os_time).all()
                                 for e in ["lr", "dr", "efs"]),
        "efs unchanged": (fix.efs_event.values == old.efs_event.values).all(),
        "os unchanged": (fix.os_event.values == old.os_event.values).all(),
        "HPV unchanged": (fix.HPV.values == old.HPV.values).all(),
        "HPV=1 protective": float(np.exp(CoxPHFitter().fit(
            fix[["os_time", "os_event", "HPV"]], "os_time", "os_event").params_["HPV"])) < 1,
    }
    changed_cols = sorted({c["column"] for c in changes})
    allowed = (set(REPLACE) | {f"{y}_{e}" for y in HORIZONS for e in REPLACE}
               | {f"{y}_{e.replace('_event', '')}_time" for y in HORIZONS for e in REPLACE})
    checks["only intended columns changed"] = set(changed_cols) <= allowed
    raw_fix = pd.read_csv(OUT, dtype=str, keep_default_na=False)
    checks["derived 2yr/3yr LR/DM columns follow the rule"] = all(
        horizon_cells(raw_fix[f"{b}_time"].iloc[i], int(raw_fix[f"{b}_event"].iloc[i]), H)[0]
        == raw_fix[f"{y}_{b}_event"].iloc[i].strip()
        for b in ("lr", "dr") for y, H in HORIZONS.items() for i in range(len(raw_fix)))

    print(f"wrote {OUT.name}   ({fix.shape[0]} rows x {fix.shape[1]} cols)")
    print(f"wrote {LOG.name}   ({len(changes)} changed cells)\n")
    print("changed cells per column:")
    for c in changed_cols:
        print(f"   {c:<16} {sum(x['column'] == c for x in changes)}")
    print("\nvalidation:")
    for k, v in checks.items():
        print(f"   {'PASS' if v else 'FAIL'}  {k}")
    if not all(checks.values()):
        raise SystemExit("\nVALIDATION FAILED -- do not use the output")
    print("\nevent counts in the corrected file:")
    for e in ["os", "efs", "lr", "dr"]:
        n = int(fix[f"{e}_event"].sum())
        print(f"   {e:<4} {n:>3}  ({100 * n / len(fix):.1f}%)")


if __name__ == "__main__":
    main()
