"""Synthetic Radcure / Maastro cohorts for the smoke test (tests/smoke_test.sh). NO real data.

Writes, under --out (a "data" folder laid out as config.yaml expects):
    new_radcure_data.csv              "original" Radcure table: composite LR/DM (death counted as an event) and
                                      RFS times set to the death time for patients who recurred and later died
    new_maastro_data.csv              "original" Maastro table: composite LR/DM
    maastro_pure_lr_dm_events.csv     second Maastro export with the pure LR / DM events (exactly 31 / 27, the
                                      counts build_maastro_corrected.py checks)
    imagesTs_radcure/, imagesTs_maastro/   one NIfTI per patient: 160 x 160 x 64 voxels, 1 x 1 x 3 mm, LPS, float32
                                      in [0, 1] (= HU windowed to [-200, 275]), as the pre-processed study scans
    predsTs_radcure/, predsTx_maastro/     label maps on the same grid (1 = "tumour", 2 = "node")
The corrected tables are then made by data_preparation/build_{radcure,maastro}_corrected.py, as for the real data.

Patients are random: identifiers SYNR_<n>_0000.nii.gz / SYNM_<n>_0000.nii.gz, clinical variables drawn from simple
distributions, survival times from a proportional-hazards model of the clinical variables, scans = a body ellipse
with smooth random texture and a "tumour" sphere whose size follows T stage.

    python tests/make_synthetic_data.py --out <folder> [--n-radcure 120] [--n-maastro 80] [--seed 0]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import zoom

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data_preparation"))
from derived_cols import HORIZONS, horizon_cells  # noqa: E402

SHAPE = (160, 160, 64)                       # stored scan size (x, y, z), spacing 1 x 1 x 3 mm
AFFINE = np.diag([-1.0, -1.0, 3.0, 1.0])     # orientation codes L, P, S (as the study scans)
N7_TO_N8_HPV_POS = {"0": "N0", "1": "N1", "2A": "N1", "2B": "N1", "2C": "N2", "3": "N3"}
N_ORD = {"0": 0, "1": 1, "2": 2, "2A": 2, "2B": 2, "2C": 2, "3": 3}


def clinical(rng, n: int, cohort: str) -> pd.DataFrame:
    hpv = (rng.random(n) < 0.65).astype(int)
    T = rng.choice([1, 2, 3, 4], n, p=[0.2, 0.35, 0.25, 0.2])
    d = pd.DataFrame({"age": np.round(np.clip(rng.normal(62, 9, n), 35, 90), 1), "HPV": hpv, "T_Stage": T,
                      "sex": (rng.random(n) < 0.8).astype(int), "smoking": (rng.random(n) < 0.6).astype(int)})
    if cohort == "Radcure":                  # raw N as recorded; N_Stage_8 = the converted AJCC category
        N = rng.choice(["0", "1", "2A", "2B", "2C", "3"], n, p=[0.15, 0.15, 0.15, 0.3, 0.15, 0.1])
        d["N"] = N
        d["N_Stage"] = [N_ORD[x] for x in N]
        d["N_Stage_8"] = [N7_TO_N8_HPV_POS[x] if h else "N" + x for x, h in zip(N, hpv)]
        n8 = np.array([N_ORD[x[1:]] for x in d["N_Stage_8"]])
    else:                                    # raw N already the harmonised AJCC category (no a/b/c for p16-positive)
        N = np.where(hpv == 1, rng.choice(["0", "1", "2", "3"], n, p=[0.1, 0.5, 0.3, 0.1]),
                     rng.choice(["0", "1", "2a", "2b", "2c", "3"], n, p=[0.1, 0.2, 0.15, 0.3, 0.15, 0.1]))
        d["N"] = N
        d["N_Stage"] = [{"0": 0, "1": 1, "2": 2, "3": 3}.get(x, 3) for x in N]
        n8 = np.array([N_ORD[x.upper()] for x in N])
    stage_pos = np.where((T >= 4) | (n8 >= 3), "III", np.where((T == 3) | (n8 == 2), "II", "I"))
    stage_neg = np.where(n8 >= 3, "IVB", np.where((T >= 3) | (n8 >= 1), "IVA", "III"))
    d["AJCC_Stage_8"] = np.where(hpv == 1, stage_pos, stage_neg)
    d["_lp"] = 0.35 * (T - 2.5) + 0.35 * (n8 - 1.5) - 1.0 * hpv + 0.02 * (d["age"] - 62)
    return d


def outcomes(rng, d: pd.DataFrame, n_lr: int | None = None, n_dr: int | None = None) -> pd.DataFrame:
    """Pure failure times, then the composite / original columns. n_lr, n_dr = exact pure event counts."""
    n, lp = len(d), d["_lp"].values
    death = rng.exponential(2500 * np.exp(-lp))
    cens = rng.uniform(300, 3000, n)
    os_time = np.maximum(np.round(np.minimum(death, cens)), 30).astype(int)
    os_event = (death <= cens).astype(int)
    pure = {}
    for e, scale, k in (("lr", 3500, n_lr), ("dr", 4000, n_dr)):
        lat = rng.exponential(scale * np.exp(-lp))
        ratio = lat / os_time
        if k is None:
            ev = ratio < 1
        else:                                            # exactly k events: the k earliest relative to follow-up
            ev = np.zeros(n, bool)
            ev[np.argsort(ratio)[:k]] = True
        t = np.where(ratio < 1, np.round(lat), np.round(os_time * rng.uniform(0.2, 0.9, n)))
        t = np.clip(t, 1, os_time - 1).astype(int)       # a failure is always before the end of follow-up
        pure[e] = (ev.astype(int), np.where(ev, t, os_time))
    d = d.copy()
    d["os_event"], d["os_time"] = os_event, os_time
    for e in ("lr", "dr"):                               # composite: death without failure counts as an event
        ev, t = pure[e]
        d[f"{e}_event"] = (ev | os_event).astype(int)
        d[f"{e}_time"] = np.where(ev == 1, t, os_time)
        d[f"_pure_{e}_event"], d[f"_pure_{e}_time"] = ev, t
    first = np.minimum.reduce([np.where(pure["lr"][0] == 1, pure["lr"][1], np.inf),
                               np.where(pure["dr"][0] == 1, pure["dr"][1], np.inf),
                               np.where(os_event == 1, os_time, np.inf)])
    d["efs_event"] = np.isfinite(first).astype(int)
    d["efs_time"] = np.where(np.isfinite(first), first, os_time).astype(int)
    return d


def add_derived(d: pd.DataFrame) -> pd.DataFrame:
    """2yr / 3yr columns with the data files' own rule (derived_cols.horizon_cells)."""
    d = d.copy()
    for y, H in HORIZONS.items():
        for e in ("os", "efs", "lr", "dr"):
            cells = [horizon_cells(str(t), int(v), H) for t, v in zip(d[f"{e}_time"], d[f"{e}_event"])]
            d[f"{y}_{e}_event"] = [c[0] for c in cells]
            d[f"{y}_{e}_time"] = [c[1] for c in cells]
    return d


def scan(rng, T: int):
    """(image, label map): body ellipse with smooth texture (fat ~0.21, muscle ~0.52), a sphere per T stage."""
    x, y, z = np.meshgrid(*[np.arange(s) for s in SHAPE], indexing="ij")
    cx, cy = 80 + rng.integers(-5, 6), 80 + rng.integers(-5, 6)
    body = ((x - cx) / rng.uniform(55, 68)) ** 2 + ((y - cy) / rng.uniform(45, 58)) ** 2 <= 1
    tex = zoom(rng.random((10, 10, 4)), (16, 16, 16), order=1)[:SHAPE[0], :SHAPE[1], :SHAPE[2]]
    img = np.where(body, np.where(tex < 0.45, 0.21, 0.52), 0.0)
    lab = np.zeros(SHAPE, np.uint8)
    for value, r, off in ((1, 2 + 2 * T, 0), (2, 5, 25)):
        px, py, pz = cx + off + rng.integers(-10, 11), cy + rng.integers(-10, 11), 32 + rng.integers(-10, 11)
        ball = ((x - px) ** 2 + (y - py) ** 2 + ((z - pz) * 3) ** 2 <= r ** 2) & body
        img[ball] = 0.62
        lab[ball] = value
    return np.round(img * 32) / 32, lab                  # coarse steps: compresses well


def write(out: Path, cohort: str, d: pd.DataFrame, rng) -> None:
    img_dir = out / ("imagesTs_radcure" if cohort == "Radcure" else "imagesTs_maastro")
    msk_dir = out / ("predsTs_radcure" if cohort == "Radcure" else "predsTx_maastro")
    img_dir.mkdir(parents=True, exist_ok=True)
    msk_dir.mkdir(parents=True, exist_ok=True)
    for name, T in zip(d["img_nn_id"], d["T_Stage"]):
        img, lab = scan(rng, int(T))
        nib.save(nib.Nifti1Image(img.astype(np.float32), AFFINE), img_dir / name)
        nib.save(nib.Nifti1Image(lab, AFFINE), msk_dir / name.replace("_0000.nii.gz", ".nii.gz"))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True, help="data folder to create")
    ap.add_argument("--n-radcure", type=int, default=120)
    ap.add_argument("--n-maastro", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.n_maastro < 70:
        raise SystemExit("--n-maastro must be >= 70 (the Maastro builder checks for exactly 31 LR / 27 DM events)")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    base = ["img_nn_id", "age", "HPV", "T_Stage", "N_Stage", "sex", "smoking", "N"]
    ends = ["os_event", "os_time", "efs_event", "efs_time", "lr_event", "lr_time", "dr_event", "dr_time"]
    derived = [f"{y}_{e}_{k}" for y in ("3yr", "2yr") for e in ("os", "efs", "lr", "dr") for k in ("event", "time")]

    # ---- Radcure: original file with composite LR/DM and late RFS times for recurrence-then-death
    r = outcomes(rng, clinical(rng, a.n_radcure, "Radcure"))
    r.insert(0, "img_nn_id", [f"SYNR_{i:04d}_0000.nii.gz" for i in range(1, len(r) + 1)])
    late = (r.os_event == 1) & ((r._pure_lr_event == 1) | (r._pure_dr_event == 1))
    r.loc[late, "efs_time"] = r.loc[late, "os_time"]
    r = add_derived(r)
    r[base + ends + derived + ["N_Stage_8", "AJCC_Stage_8"]].to_csv(out / "new_radcure_data.csv", index=False)

    # ---- Maastro: original file (composite LR/DM, correct RFS) + the pure-event export
    m = outcomes(rng, clinical(rng, a.n_maastro, "Maastro"), n_lr=31, n_dr=27)
    m.insert(0, "img_nn_id", [f"SYNM_{i:04d}_0000.nii.gz" for i in range(1, len(m) + 1)])
    m = add_derived(m)
    m[base[:-1] + ends + ["N", "AJCC_Stage_8"] + derived].to_csv(out / "new_maastro_data.csv", index=False)
    new = m[base[:-1] + ends].copy()
    for e in ("lr", "dr"):        # the export's own censoring times differ (later than os_time); events match
        new[f"{e}_event"] = m[f"_pure_{e}_event"]
        new[f"{e}_time"] = np.where(m[f"_pure_{e}_event"] == 1, m[f"_pure_{e}_time"], m.os_time + 30)
    new["HPV"] = 1 - m["HPV"]     # coded the other way round in that export (not used from it)
    new.to_csv(out / "maastro_pure_lr_dm_events.csv", index=False)

    for c, d in (("Radcure", r), ("Maastro", m)):
        write(out, c, d, rng)
        print(f"{c}: {len(d)} synthetic patients, events OS {int(d.os_event.sum())}, RFS {int(d.efs_event.sum())}, "
              f"pure LR {int(d._pure_lr_event.sum())}, pure DM {int(d._pure_dr_event.sum())}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
