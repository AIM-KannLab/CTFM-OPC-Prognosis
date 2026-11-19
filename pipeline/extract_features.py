"""CT-FM feature extraction, cached to disk -- one feature set per preprocessing version.

The CT-FM backbone is frozen, so the 512-d features are endpoint-independent:
extract once per cohort and reuse for all four endpoints and every re-run.

Two preprocessing versions -- choose with --preproc when extracting, and select the
set an analysis reads with the A1_FEATURES env variable (see cohort.py):

  published  verbatim from few_shot_single_institution_running.py:127-135; reproduces
             the submitted manuscript. Its defect: the image files on disk are ALREADY
             windowed to [-200, 275] HU and scaled to [0, 1] by the offline
             preprocessing, and CT-FM's transform -- which expects raw HU -- is applied
             on top, squeezing every scan into [0.3333, 0.3337]. The crop-to-body step
             (CropForeground) then never crops.                         -> features/

  hu         converts the stored values back to Hounsfield units first, so CT-FM gets
             the input its model card specifies (raw HU, then -1024..2048 -> 0..1):
                 0 < x < 1   HU = 475 x - 200   exact (window width 475 HU)
                 x == 0      HU = -1024         stored 0 = at or below -200 HU: air
                 x == 1      HU = 275           stored 1 = at or above 275 HU: bone
             Bone, teeth and contrast were clipped when the files were made, so their
             true values are lost -- the one remaining difference from CT-FM's
             pretraining input. With air now at 0 the crop-to-body step actually crops,
             so scans differ in size and are processed one at a time.   -> features_hu/

The [-200, 275] HU window is the one stated in the manuscript Methods, and is
confirmed from the images themselves: under it, the intensity histogram peaks fall
on fat (about -100 HU) and muscle (about +45 HU).

    CUDA_VISIBLE_DEVICES=0 python extract_features.py --preproc hu
    CUDA_VISIBLE_DEVICES=0 python extract_features.py --preproc published --force   # rebuild old set
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from monai.transforms import (
    Compose, LoadImage, EnsureType, Orientation, ScaleIntensityRange, CropForeground, Lambda,
)
from lighter_zoo import SegResEncoder

from cohort import COHORTS, FEATURE_DIRS, FEATURE_VERSION, build_cohort
from ctfm_config import ctfm_encoder

# HU window the stored image files were made with (manuscript Methods; verified from
# the intensity histograms).
WINDOW_LO, WINDOW_HI = -200.0, 275.0
AIR_HU = -1024.0
# Scans are a fixed 64x160x160 under `published`, so they can be batched; under `hu`
# the crop-to-body step gives each scan its own size, so they must go one at a time.
DEFAULT_BATCH = {"published": 8, "hu": 1}


def to_hu(img):
    """Undo the offline window: stored [0, 1] values -> Hounsfield units."""
    hu = img * (WINDOW_HI - WINDOW_LO) + WINDOW_LO     # exact for 0 < x < 1
    hu[img <= 0] = AIR_HU                              # air / background
    hu[img >= 1] = WINDOW_HI                           # bone: true value already lost
    return hu


def build_transform(preproc: str) -> Compose:
    # CT-FM model card: raw HU, clipped to [-1024, 2048] HU and scaled to [0, 1].
    ctfm_scale = ScaleIntensityRange(a_min=-1024, a_max=2048, b_min=0, b_max=1, clip=True)
    if preproc == "published":
        # Verbatim from few_shot_single_institution_running.py:127-135.
        return Compose([LoadImage(ensure_channel_first=True), EnsureType(),
                        Orientation(axcodes="SPL"), ctfm_scale,
                        CropForeground(allow_smaller=True)])
    if preproc == "hu":
        return Compose([LoadImage(ensure_channel_first=True), EnsureType(),
                        Lambda(to_hu),                 # the only difference from `published`
                        Orientation(axcodes="SPL"), ctfm_scale,
                        CropForeground(allow_smaller=True)])
    raise ValueError(f"unknown preprocessing {preproc!r}; expected one of {list(FEATURE_DIRS)}")


class ImageDataset(Dataset):
    """Yields preprocessed volumes in a fixed order matching the input dataframe."""

    def __init__(self, paths, preproc: str):
        self.paths = [str(p) for p in paths]
        self.transform = build_transform(preproc)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        return self.transform(self.paths[idx])


def extract(cohort: str, preproc: str, batch_size: int | None = None, num_workers: int = 4,
            force: bool = False, out_dir: Path | None = None) -> Path:
    out_dir = Path(out_dir) if out_dir else FEATURE_DIRS[preproc]
    out_path = out_dir / f"features_{cohort.lower()}.npz"
    if out_path.exists() and not force:
        raise SystemExit(f"{out_path} already exists -- refusing to overwrite it "
                         f"(it may be the feature set behind existing results). "
                         f"Pass --force to replace it.")
    batch_size = batch_size or DEFAULT_BATCH[preproc]
    if preproc == "hu" and batch_size != 1:
        raise SystemExit("--preproc hu needs --batch_size 1: the crop-to-body step gives "
                         "each scan its own size, so scans cannot be stacked into batches")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    where = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    print(f"\n=== {cohort} === preproc={preproc}  device={where}  batch={batch_size}")

    # Endpoint choice is irrelevant for features -- the backbone is frozen and the
    # full cohort is extracted, not a split. 'os' is used only to build the table.
    df = build_cohort(cohort, "os")
    print(f"  {len(df)} patients -> {out_path}")

    backbone = ctfm_encoder().to(device)      # model.ctfm_weights (project-lighter/ct_fm_feature_extractor)
    backbone.eval()
    for m in backbone.modules():
        if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
            m.eval()

    loader = DataLoader(
        ImageDataset(df["path"].tolist(), preproc),
        batch_size=batch_size,
        shuffle=False,          # order must match df -- keys are assigned positionally
        num_workers=num_workers,
    )

    feats, t0 = [], time.time()
    report_every = max(1, 160 // batch_size)          # progress every ~160 patients
    with torch.no_grad():
        for i, imgs in enumerate(loader):
            imgs = imgs.to(device)
            out = backbone(imgs)[-1]
            feats.append(F.adaptive_avg_pool3d(out, 1).flatten(1).cpu().numpy())
            if (i + 1) % report_every == 0:
                done = min((i + 1) * batch_size, len(df))
                print(f"  {done}/{len(df)}  ({time.time() - t0:.0f}s)")

    features = np.concatenate(feats, axis=0)
    if len(features) != len(df):
        raise RuntimeError(f"feature count {len(features)} != patient count {len(df)}")
    if not np.isfinite(features).all():
        raise RuntimeError("non-finite values in the extracted features")

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, keys=df["key"].values,
                        features=features.astype(np.float32), preproc=preproc)
    print(f"  saved {features.shape} -> {out_path}  ({time.time() - t0:.0f}s)")
    return out_path


def load_features(cohort: str, version: str | None = None) -> dict:
    """Return {patient_key: 512-d feature vector} for a cohort.

    `version` defaults to A1_FEATURES (cohort.FEATURE_VERSION), so every analysis
    script reads the feature set selected for the run without code changes.
    """
    version = version or FEATURE_VERSION
    path = FEATURE_DIRS[version] / f"features_{cohort.lower()}.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- run: "
                                f"python extract_features.py --preproc {version}")
    d = np.load(path, allow_pickle=True)
    stored = str(d["preproc"]) if "preproc" in d.files else "published"
    if stored != version:
        raise ValueError(f"{path} holds '{stored}' features, but '{version}' was requested")
    return dict(zip(d["keys"], d["features"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--preproc", required=True, choices=list(FEATURE_DIRS),
                    help="hu = corrected (recommended); published = as in the manuscript")
    ap.add_argument("--cohorts", nargs="+", default=list(COHORTS))
    ap.add_argument("--batch_size", type=int, default=None,
                    help="default: 8 for published, 1 for hu")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--force", action="store_true", help="overwrite an existing feature file")
    args = ap.parse_args()
    for c in args.cohorts:
        extract(c, args.preproc, args.batch_size, args.num_workers, args.force)
