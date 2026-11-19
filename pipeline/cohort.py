"""Cohort construction and split reproduction for the A1 revision analysis.

Everything here mirrors the published pipeline so that train/val/test membership
is identical to the runs that produced the submitted manuscript. Two details are
load-bearing and must not be "cleaned up":

1. `split_dataset_ratio` is a verbatim port of
   `working_space/few_shot_single_institution_running.py:512-574`. It stratifies on
   the endpoint's own `event` column, so the test set DIFFERS between endpoints.
2. Radcure drops rows whose resolved image path is missing; Maastro does not.
   That asymmetry is inherited from `build_radcure` / `build_maastro` in the same
   file (lines 440-490). Currently nothing is dropped from either cohort, but the
   asymmetry is preserved so the split cannot silently diverge.

Unlike the published builders, which return only [path, time, event], these carry
an explicit patient `key` plus the clinical covariates needed for the A1 baseline.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from ctfm_config import CFG, cache_dir, data_path, outputs_dir

# Input files and folders come from the configuration (config.yaml + CTFM_* environment
# variables; see ctfm_config.py), not from fixed locations.

# Which version of the clinical data to analyse. Set with the A1_DATA env variable.
#   corrected (default)  LR / DM are PURE failure endpoints (death censored).
#                        Built by build_radcure_corrected.py / build_maastro_corrected.py.
#   original             the files behind the submitted manuscript, where LR / DM are
#                        COMPOSITE (death counted as an event). Use only to reproduce
#                        the published numbers.
# The files: data.{radcure,maastro}_csv_corrected and data.{radcure,maastro}_csv (config.yaml).
DATA_VERSION = os.environ.get("A1_DATA") or "corrected"        # empty = unset, as in bash
_CSV = {
    "corrected": {"Radcure": data_path("data.radcure_csv_corrected"),
                  "Maastro": data_path("data.maastro_csv_corrected")},
    "original":  {"Radcure": data_path("data.radcure_csv"),
                  "Maastro": data_path("data.maastro_csv")},
}
if DATA_VERSION not in _CSV:
    raise ValueError(f"A1_DATA must be one of {list(_CSV)}, got {DATA_VERSION!r}")

_HERE = Path(__file__).parent
# Where a run's outputs go -- result folders, SegRes networks, SPECTRE features. Set with the
# A1_OUTPUT_ROOT env variable (a relative path is taken from outputs.dir); default outputs.dir
# (config.yaml). run_pipeline.sh uses <outputs.dir>/<run id>_study/, so one run
# sits in one folder. Inputs (CT-FM features, preprocessed scans) stay in cache.dir and are shared.
OUTPUT_ROOT = outputs_dir() / (os.environ.get("A1_OUTPUT_ROOT") or ".")

# Which CT images. Set with the A1_IMAGES env variable.
#   original (default)  data.radcure_images / data.maastro_images (config.yaml) -- the files of the
#                       submitted paper and of every revision run: windowed to [-200, 275] HU and
#                       scaled to [0, 1], one fixed scale for all scans.
# (The revision code had a second, unpublished image set here; it is not part of this repository.)
IMAGES = os.environ.get("A1_IMAGES") or "original"
IMAGE_DIRS = {
    "original": {"Radcure": data_path("data.radcure_images"), "Maastro": data_path("data.maastro_images")},
}
if IMAGES not in IMAGE_DIRS:
    raise ValueError(f"A1_IMAGES must be one of {list(IMAGE_DIRS)}, got {IMAGES!r}")
_IMG_TAG = "" if IMAGES == "original" else f"_{IMAGES}"

# Which CT-FM feature set to analyse. Set with the A1_FEATURES env variable.
#   published (default)  features as the submitted manuscript computed them. The image
#                        files are already windowed to [-200, 275] HU and scaled to
#                        [0, 1]; CT-FM's own transform (which expects raw HU) is then
#                        applied on top, squeezing every scan into [0.3333, 0.3337].
#   hu                   the stored values are first converted back to Hounsfield units,
#                        so CT-FM receives the input its model card specifies. Bone
#                        (>= 275 HU) was clipped when the files were made and stays at 275.
# Produced by:  python extract_features.py --preproc <version>
FEATURE_VERSION = os.environ.get("A1_FEATURES") or "published"  # empty = unset, as in bash
FEATURE_DIRS = {"published": cache_dir() / f"features{_IMG_TAG}", "hu": cache_dir() / f"features_hu{_IMG_TAG}"}
if FEATURE_VERSION not in FEATURE_DIRS:
    raise ValueError(f"A1_FEATURES must be one of {list(FEATURE_DIRS)}, got {FEATURE_VERSION!r}")
FEATURE_DIR = FEATURE_DIRS[FEATURE_VERSION]
# SPECTRE features (extract_spectre_features.py), one set per preprocessing version
SPECTRE_FEATURE_DIRS = {v: OUTPUT_ROOT / f"features_spectre_{v}{_IMG_TAG}" for v in FEATURE_DIRS}

# Which imaging model the analysis uses. Set with the A1_IMAGING env variable.
#   ctfm_st (default)  frozen CT-FM features + ridge Cox -- the manuscript's model
#   segres_ft          SegRes-FT: the same SegResEncoder architecture trained FROM SCRATCH
#                      (random init) with an MLP Cox head, configured as the "SegRes-FT"
#                      entry of few_shot_single_institution_*_running.py. Its input is the
#                      A1_FEATURES preprocessing. The networks are trained by
#                      train_segres.py and read back from SEGRES_DIR.
IMAGING = os.environ.get("A1_IMAGING") or "ctfm_st"             # empty = unset, as in bash
IMAGING_LABEL = {"ctfm_st": "CT-FM-ST", "segres_ft": "SegRes-FT"}
IMAGING_SHORT = {"ctfm_st": "CT-FM", "segres_ft": "SegRes-FT"}  # as in "Clinical + CT-FM"
if IMAGING not in IMAGING_LABEL:
    raise ValueError(f"A1_IMAGING must be one of {list(IMAGING_LABEL)}, got {IMAGING!r}")
# Which nodal-stage scale the clinical models use. Set with the A1_NSTAGE env variable.
#   ajcc8 (default)  N_Stage is the harmonised AJCC (p16-specific) category in BOTH cohorts,
#                    collapsed to N0=0, N1=1, N2 (any sub-category)=2, N3=3. See _nstage_ajcc8.
#   raw              each CSV's own N_Stage column, which is NOT the same scale in the two
#                    cohorts: Radcure's uses another AJCC version, and Maastro's miscodes N2a/N2b/N2c as 3.
#                    A model trained on one cohort therefore cannot be applied to the other.
#                    Only for reproducing runs made before the harmonisation (no _n8 in the
#                    folder name).
NSTAGE = os.environ.get("A1_NSTAGE") or "ajcc8"                 # empty = unset, as in bash
if NSTAGE not in ("ajcc8", "raw"):
    raise ValueError(f"A1_NSTAGE must be ajcc8 or raw, got {NSTAGE!r}")
# SegRes-FT only. fp32 = as the original scripts (default); bf16 = autocast, ~2x faster.
SEGRES_PRECISION = os.environ.get("A1_SEGRES_PRECISION") or "fp32"
if SEGRES_PRECISION not in ("fp32", "bf16"):
    raise ValueError(f"A1_SEGRES_PRECISION must be fp32 or bf16, got {SEGRES_PRECISION!r}")
# Maximum training epochs; 40 in the original configuration. Change only for smoke tests.
SEGRES_EPOCHS = int(os.environ.get("A1_SEGRES_EPOCHS") or 40)
# Which from-scratch SegRes model train_segres.py trains. Set with A1_SEGRES_VARIANT.
#   ft  (default)  SegRes-FT: MLP head, BatchNorm trained, plus out-of-fold networks for the
#                  combined model -- read by run_a1.py / run_external.py (A1_IMAGING=segres_ft)
#   e2e            SegRes-E2E, the submitted paper's from-scratch baseline: one linear layer on
#                  the pooled features, BatchNorm frozen. Imaging-only, read by run_baselines.py:
#                  5 seeds at EVERY training size including full shot, no out-of-fold networks.
SEGRES_VARIANT = os.environ.get("A1_SEGRES_VARIANT") or "ft"
if SEGRES_VARIANT not in ("ft", "e2e"):
    raise ValueError(f"A1_SEGRES_VARIANT must be ft or e2e, got {SEGRES_VARIANT!r}")
# Preprocessed scans (one array per cohort), shared by every SegRes-FT network.
VOLUME_DIR = cache_dir() / "volumes" / f"{FEATURE_VERSION}{_IMG_TAG}"
# Trained SegRes-FT networks and their outputs, one folder per training set.
SEGRES_DIR = (OUTPUT_ROOT / f"segres_{SEGRES_VARIANT}" / f"{DATA_VERSION}_{FEATURE_VERSION}{_IMG_TAG}_{SEGRES_PRECISION}"
              f"{'' if SEGRES_EPOCHS == 40 else f'_e{SEGRES_EPOCHS}'}")


def _imaging_tag() -> str:
    """'' for CT-FM-ST; '_segresft' (+ non-default precision / epochs) for SegRes-FT."""
    if IMAGING == "ctfm_st":
        return ""
    tag = "_segresft"
    if SEGRES_PRECISION != "fp32":
        tag += f"-{SEGRES_PRECISION}"
    if SEGRES_EPOCHS != 40:
        tag += f"-e{SEGRES_EPOCHS}"
    return tag


def _version_tag() -> str:
    """Data version, plus the feature version when it is not the default, plus `_n8` for
    harmonised nodal stage, plus the imaging model when it is not CT-FM-ST.

    `_n8` marks the harmonised runs rather than the raw ones, so the folders of runs made
    before the harmonisation keep their names and can still be reproduced (A1_NSTAGE=raw).
    """
    base = DATA_VERSION if FEATURE_VERSION == "published" else f"{DATA_VERSION}_{FEATURE_VERSION}"
    return base + _IMG_TAG + ("_n8" if NSTAGE == "ajcc8" else "") + _imaging_tag()


_RUN_ID = re.compile(r"\d{8}_\d{6}")


def new_run_id() -> str:
    """Timestamp used as a run id: YYYYMMDD_HHMMSS."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def results_dir(name: str = "results", run: str | None = None) -> Path:
    """Timestamped folder for one run:  <name>_<data version>[_<features>]_<YYYYMMDD_HHMMSS>.

    The feature version appears only when it is not the default, so e.g.
    results_corrected_20261002_153012 (published features) and
    results_corrected_hu_20261002_160501 (HU features) never mix. Likewise SegRes-FT
    runs are results_corrected_hu_segresft_<run>.

    Every run gets its own folder, so no run can overwrite or satisfy the resume check
    of another. Pass the same `run` id to resume a run or add cells to it (needed when
    several tmux panes contribute to one run). The pre-timestamp folders (`results/`,
    `results_corrected/`, ...) are left untouched.
    """
    return OUTPUT_ROOT / f"{name}_{_version_tag()}_{run or new_run_id()}"


def latest_results_dir(name: str = "results") -> Path:
    """Newest timestamped run folder for this data + feature version."""
    prefix = f"{name}_{_version_tag()}_"
    hits = sorted(p for p in OUTPUT_ROOT.glob(prefix + "*")
                  if p.is_dir() and _RUN_ID.fullmatch(p.name[len(prefix):]))
    if not hits:
        raise FileNotFoundError(f"no {prefix}<YYYYMMDD_HHMMSS> folder in {OUTPUT_ROOT}")
    return hits[-1]


def write_run_info(folder: Path, **extra) -> None:
    """Record what this run used -- data files with md5, data version, command line.
    Written once per run folder (the first pane to start wins; later panes skip)."""
    folder.mkdir(parents=True, exist_ok=True)
    info = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "data_version": DATA_VERSION,
        "data_files": {c: {"path": str(s["csv"]),
                           "md5": hashlib.md5(Path(s["csv"]).read_bytes()).hexdigest()}
                       for c, s in COHORTS.items()},
        "feature_version": FEATURE_VERSION,
        "feature_files": {c: {"path": str(f),
                              "md5": hashlib.md5(f.read_bytes()).hexdigest() if f.exists() else None}
                          for c in COHORTS
                          # FEATURE_DIRS, not FEATURE_DIR: runner.py re-points the folder after import
                          for f in [FEATURE_DIRS[FEATURE_VERSION] / f"features_{c.lower()}.npz"]},
        "nstage": NSTAGE,
        "imaging": IMAGING,
        "images": IMAGES,
        "image_dirs": {c: str(s["img_dir"]) for c, s in COHORTS.items()},
        "output_root": str(OUTPUT_ROOT.resolve()),
        "config_file": str(CFG.file),
        "config": CFG.resolved(),
        **({"segres": {"precision": SEGRES_PRECISION, "max_epochs": SEGRES_EPOCHS,
                       "networks": str(SEGRES_DIR), "volumes": str(VOLUME_DIR)}}
           if IMAGING == "segres_ft" else {}),
        "command": " ".join(sys.argv),
        "python": sys.version.split()[0],
        "packages": _package_versions(),
        "hardware": _hardware(),
        "code_snapshot": "code_snapshot/ (the exact scripts this run used, with md5s)",
        **extra,
    }
    try:
        with open(folder / "run_info.json", "x") as fh:
            json.dump(info, fh, indent=2)
    except FileExistsError:
        pass
    _snapshot_code(folder / "code_snapshot")


_PACKAGES = ["numpy", "pandas", "scipy", "scikit-learn", "lifelines", "scikit-survival",
             "torch", "monai", "lighter_zoo", "nibabel", "matplotlib", "spectre-fm"]


def _package_versions() -> dict:
    """Installed versions, read from package metadata (nothing is imported)."""
    from importlib.metadata import PackageNotFoundError, version
    out = {}
    for p in _PACKAGES:
        try:
            out[p] = version(p)
        except PackageNotFoundError:
            out[p] = None
    return out


def _hardware() -> dict:
    import platform
    import subprocess
    gpus = None
    try:
        gpus = subprocess.run(["nvidia-smi", "--query-gpu=index,name,driver_version",
                               "--format=csv,noheader"], capture_output=True, text=True,
                              timeout=20).stdout.strip().splitlines()
    except Exception:
        pass
    return {"host": platform.node(), "cpu_count": os.cpu_count(), "gpus": gpus,
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES")}


def _snapshot_code(dest: Path) -> None:
    """Copy this folder's scripts into the run folder, once, with an md5 list -- the code
    changed during the revision, and a result is only reproducible with the code that made it."""
    import shutil
    try:
        dest.mkdir(parents=True)         # several jobs start at once: only the first copies
    except FileExistsError:
        return
    lines = []
    for f in sorted(list(_HERE.glob("*.py")) + list(_HERE.parent.glob("*.sh"))):
        shutil.copy2(f, dest / f.name)
        lines.append(f"{hashlib.md5(f.read_bytes()).hexdigest()}  {f.name}")
    (dest / "MD5SUMS").write_text("\n".join(lines) + "\n")


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
        return len(s)

    def flush(self):
        for st in self.streams:
            st.flush()


def tee_output(log_path: Path) -> None:
    """Copy everything printed (stdout + stderr) into a log inside the run folder."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(log_path, "a", buffering=1)
    sys.stdout = _Tee(sys.stdout, fh)
    sys.stderr = _Tee(sys.stderr, fh)


COHORTS = {
    "Radcure": {
        "csv": _CSV[DATA_VERSION]["Radcure"],
        "img_dir": IMAGE_DIRS[IMAGES]["Radcure"],
        "filter_missing": True,   # matches build_radcure
        # Column holding the harmonised AJCC nodal category. Radcure's own N / N_Stage use
        # another AJCC version; the file ships the converted category alongside, in N_Stage_8.
        "n8_col": "N_Stage_8",
        "n_raw_edition": 7,
    },
    "Maastro": {
        "csv": _CSV[DATA_VERSION]["Maastro"],
        "img_dir": IMAGE_DIRS[IMAGES]["Maastro"],
        "filter_missing": False,  # matches build_maastro
        # Maastro's raw N is already the harmonised AJCC category (p16-positive patients carry no a/b/c
        # sub-categories). Its derived N_Stage column is not used: it miscodes N2a/b/c as 3.
        "n8_col": "N",
        "n_raw_edition": 8,
    },
}

# Code-side endpoint names. The manuscript renames dr -> DM and efs -> RFS.
ENDPOINTS = ["os", "efs", "lr", "dr"]
ENDPOINT_PAPER_NAME = {"os": "OS", "efs": "RFS", "lr": "LR", "dr": "DM"}

CLINICAL_COLS = ["age", "sex", "HPV", "smoking", "T_Stage", "N_Stage", "AJCC_Stage_8"]

# `N_Stage` is the AJCC category on one scale across cohorts unless A1_NSTAGE=raw;
# see _nstage_ajcc8. `T_Stage` needs no harmonisation: T1-T4 are coded the same in both cohorts
# (the T4a/T4b split, which differs for p16-positive disease, is recorded by neither cohort).
# Primary spec: T+N. Sensitivity spec: AJCC instead (the AJCC stage is a deterministic
# function of T, N and HPV, so including all three is collinear).
SPEC_PRIMARY = ["age", "sex", "HPV", "smoking", "T_Stage", "N_Stage"]
SPEC_AJCC = ["age", "sex", "HPV", "smoking", "AJCC_Stage_8"]

_AJCC_MAP = {"I": 1, "II": 2, "III": 3, "IVA": 4, "IVB": 5, "IVC": 6}

# AJCC nodal category -> the ordinal the Cox models use. Sub-categories are
# collapsed, so N2a / N2b / N2c all become 2 (as the published N_Stage column already did).
_N8_ORDINAL = {"N0": 0, "N1": 1, "N2": 2, "N2A": 2, "N2B": 2, "N2C": 2, "N3": 3}
# The standard AJCC conversion for p16-POSITIVE oropharynx: N1, N2a and N2b all become
# N1 (ipsilateral nodes <= 6 cm), N2c becomes N2 (bilateral / contralateral), N3 stays N3.
# p16-negative categories are unchanged.
_N7_TO_N8_HPV_POS = {"N0": "N0", "N1": "N1", "N2A": "N1", "N2B": "N1", "N2C": "N2", "N3": "N3"}


def _n_label(v) -> str:
    """'2b' / 'N2B' / 2 -> 'N2B'. The two cohorts spell the same categories differently."""
    s = str(v).strip().upper()
    return s if s.startswith("N") else "N" + s


def _nstage_ajcc8(df: pd.DataFrame, cohort: str) -> pd.Series:
    """AJCC nodal stage as an ordinal 0-3, on one scale for both cohorts.

    The source column differs per cohort (see COHORTS[...]['n8_col']) because the cohorts
    were staged under different AJCC versions. Both are checked against the harmonised rules
    before use, so a changed data file cannot silently feed the wrong scale into a model:

      * where the raw column needs conversion (Radcure), the shipped converted column must
        equal the standard conversion of the raw category given p16 status;
      * where it is already harmonised (Maastro), no p16-positive patient may carry an
        a/b/c sub-category -- those exist only for p16-negative disease.
    """
    spec = COHORTS[cohort]
    col = spec["n8_col"]
    if col not in df.columns:
        raise ValueError(f"{cohort} CSV has no '{col}' column, needed for the AJCC nodal stage "
                         f"(set A1_NSTAGE=raw to use each cohort's own N_Stage instead)")
    n8 = df[col].map(_n_label)
    bad = sorted(set(n8[~n8.isin(_N8_ORDINAL)]))
    if bad:
        raise ValueError(f"{cohort}: unrecognised nodal categories in '{col}': {bad}")

    hpv_pos = df["HPV"].astype(float) == 1
    if spec["n_raw_edition"] == 7:
        raw = df["N"].map(_n_label)
        expect = pd.Series([_N7_TO_N8_HPV_POS[r] if p else r for r, p in zip(raw, hpv_pos)],
                           index=df.index)
        wrong = int((expect != n8).sum())
        if wrong:
            raise ValueError(f"{cohort}: '{col}' disagrees with the standard AJCC nodal "
                             f"conversion of 'N' for {wrong} patients -- the data file changed; "
                             f"check it before trusting any clinical model")
    else:
        sub = hpv_pos & n8.isin(["N2A", "N2B", "N2C"])
        if sub.any():
            raise ValueError(f"{cohort}: {int(sub.sum())} p16-positive patients have an N2a/b/c "
                             f"sub-category in '{col}', which the harmonised AJCC staging does not define -- "
                             f"that column is not harmonised; check the data file")
    return n8.map(_N8_ORDINAL).astype(int)


def resolve_path(img_dir: Path, name: str) -> Path:
    """Resolve a filename to disk, with the TS_ -> TSR_ fallback used by build_radcure."""
    p = img_dir / str(name)
    if p.exists():
        return p
    if isinstance(name, str) and name.startswith("TS_"):
        alt = img_dir / ("TSR_" + name[3:])
        if alt.exists():
            return alt
    return p


def build_cohort(cohort: str, endpoint: str) -> pd.DataFrame:
    """Build a per-patient table for one cohort and endpoint.

    Returns columns: key, path, time, event + CLINICAL_COLS.
    Row order follows the source CSV, which is what makes the split reproducible.
    """
    if cohort not in COHORTS:
        raise ValueError(f"Unknown cohort {cohort!r}; expected one of {list(COHORTS)}")
    if endpoint not in ENDPOINTS:
        raise ValueError(f"Unknown endpoint {endpoint!r}; expected one of {ENDPOINTS}")

    spec = COHORTS[cohort]
    df = pd.read_csv(spec["csv"])

    if "filename" not in df.columns and "img_nn_id" in df.columns:
        df = df.rename(columns={"img_nn_id": "filename"})
    if "filename" not in df.columns:
        raise ValueError(f"{cohort} CSV must contain 'filename' or 'img_nn_id'")
    if IMAGES != "original":   # these folders name a scan TS_0000.nii.gz, not TS_0000_0000.nii.gz
        df["filename"] = df["filename"].astype(str).str.replace(r"_\d{4}\.nii\.gz$", ".nii.gz", regex=True)

    time_col, event_col = f"{endpoint}_time", f"{endpoint}_event"
    missing = {time_col, event_col} - set(df.columns)
    if missing:
        raise ValueError(f"{cohort} CSV missing endpoint columns {missing}")
    df = df.rename(columns={time_col: "time", event_col: "event"})

    df["path"] = df["filename"].astype(str).apply(lambda f: resolve_path(spec["img_dir"], f))
    # `key` is the resolved filename on disk, which is the de-facto patient ID.
    df["key"] = df["path"].apply(lambda p: Path(p).name)

    if spec["filter_missing"]:
        exists = df["path"].apply(lambda p: Path(p).exists())
        if (~exists).any():
            print(f"  [{cohort}] dropping {int((~exists).sum())} rows with no image on disk")
        df = df[exists].reset_index(drop=True)

    df["AJCC_Stage_8"] = df["AJCC_Stage_8"].map(_AJCC_MAP)
    if NSTAGE == "ajcc8":
        df["N_Stage"] = _nstage_ajcc8(df, cohort)

    out = df[["key", "path", "time", "event"] + CLINICAL_COLS].reset_index(drop=True)

    if out["key"].duplicated().any():
        raise ValueError(f"{cohort}: duplicate patient keys - the feature join would be ambiguous")
    n_na = out[CLINICAL_COLS].isna().sum().sum()
    if n_na:
        raise ValueError(f"{cohort}: {n_na} missing clinical values; expected none")
    return out


def split_dataset_ratio(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Verbatim port of few_shot_single_institution_running.py:512-574.

    Stratified on `event`. Do not change the seed or the two-stage structure --
    this is what reproduces the published test sets.
    """
    if not np.isclose(train_frac + val_frac + test_frac, 1.0):
        raise ValueError("Fractions must sum to 1.0")

    primary = StratifiedShuffleSplit(
        n_splits=1, test_size=val_frac + test_frac, random_state=seed
    )
    train_idx, temp_idx = next(primary.split(df, df["event"]))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    temp_df = df.iloc[temp_idx].reset_index(drop=True)

    test_ratio = test_frac / (val_frac + test_frac)
    secondary = StratifiedShuffleSplit(
        n_splits=1, test_size=test_ratio, random_state=seed
    )
    val_idx, test_idx = next(secondary.split(temp_df, temp_df["event"]))
    val_df = temp_df.iloc[val_idx].reset_index(drop=True)
    test_df = temp_df.iloc[test_idx].reset_index(drop=True)

    return train_df, val_df, test_df
