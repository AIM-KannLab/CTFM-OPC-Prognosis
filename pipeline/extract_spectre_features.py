"""SPECTRE features with the HU-corrected input -- for the SPECTRE baseline (run_baselines.py).

SPECTRE (spectre-fm, model 'spectre-large-pretrained') is a frozen ViT-Large CT foundation
model. Its pipeline, from working_space/SPECTRE_single_institution_{precompute_patches,
benchmark}.py:
    HU clipped to [-1000, 1000] -> [0, 1]; orientation RAS; resampled to 0.75 x 0.75 x 1.5 mm;
    padded / cropped to 384 x 384 x 256; cut into a 3 x 3 x 4 grid of 128 x 128 x 64 patches
    (36 per scan); backbone + feature combiner (grid 3 x 3 x 4) -> tokens; the linear Cox head
    then averages the tokens into one vector.
The submitted runs applied that pipeline to images that are ALREADY windowed to [-200, 275]
HU and scaled to [0, 1], so every scan reached SPECTRE squeezed into [0.5, 0.5005] -- the
cached patches in working_space/data/spectre_patches_cache/ confirm it. Here the stored
values are first converted back to HU with extract_features.to_hu, exactly as for the
revision's CT-FM features (--preproc hu). HU outside [-200, 275] was clipped when the image
files were made and cannot be recovered, for SPECTRE as for CT-FM.

The backbone is frozen and has no active dropout (all of its dropout layers have rate 0), so
one pass per scan gives exactly the features the original training loop recomputed every
epoch. Only the token-averaged vector per scan is stored -- not the 23 GB of patches.

    CUDA_VISIBLE_DEVICES=3 python extract_spectre_features.py --preproc hu
        -> features_spectre_hu/features_{radcure,maastro}.npz   (keys, features, preproc)
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from monai.transforms import (Compose, EnsureType, Lambda, LoadImage, Orientation,
                              ResizeWithPadOrCrop, ScaleIntensityRange, Spacing)
from torch.utils.data import DataLoader, Dataset

from cohort import COHORTS, FEATURE_DIRS, SPECTRE_FEATURE_DIRS, build_cohort
from extract_features import to_hu

TARGET = (384, 384, 256)
PATCH = (128, 128, 64)
GRID = (3, 3, 4)                 # 3 x 3 x 4 = 36 patches


def build_transform(preproc: str) -> Compose:
    steps = [LoadImage(image_only=True, ensure_channel_first=True), EnsureType()]
    if preproc == "hu":
        steps.append(Lambda(to_hu))                    # the only difference from the original
    steps += [ScaleIntensityRange(a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0, clip=True),
              Orientation(axcodes="RAS"),
              Spacing(pixdim=(0.75, 0.75, 1.5), mode="bilinear"),
              ResizeWithPadOrCrop(spatial_size=TARGET)]
    return Compose(steps)


def to_patches(img: torch.Tensor) -> torch.Tensor:
    """(C, 384, 384, 256) -> (36, C, 128, 128, 64), in the original's i, j, k order."""
    return torch.stack([img[:, i * PATCH[0]:(i + 1) * PATCH[0],
                            j * PATCH[1]:(j + 1) * PATCH[1],
                            k * PATCH[2]:(k + 1) * PATCH[2]]
                        for i in range(GRID[0]) for j in range(GRID[1]) for k in range(GRID[2])])


class Scans(Dataset):
    def __init__(self, paths, preproc):
        self.paths = [str(p) for p in paths]
        self.tf = build_transform(preproc)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return to_patches(torch.as_tensor(self.tf(self.paths[i]), dtype=torch.float32))


def extract(cohort: str, preproc: str, workers: int, force: bool) -> Path:
    from spectre import MODEL_CONFIGS, SpectreImageFeatureExtractor
    out = SPECTRE_FEATURE_DIRS[preproc] / f"features_{cohort.lower()}.npz"
    if out.exists() and not force:
        raise SystemExit(f"{out} already exists -- pass --force to replace it")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("!! no GPU visible: ViT-Large on the CPU will be very slow")
    df = build_cohort(cohort, "os")                    # every patient; endpoint irrelevant
    print(f"\n=== {cohort} === preproc={preproc}  {len(df)} scans  device="
          f"{torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu'}  -> {out}", flush=True)
    model = SpectreImageFeatureExtractor.from_config(MODEL_CONFIGS["spectre-large-pretrained"])
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad = False
    loader = DataLoader(Scans(df["path"].tolist(), preproc), batch_size=1, shuffle=False,
                        num_workers=workers)               # order must match df: keys are positional
    feats, t0 = [], time.time()
    with torch.no_grad():
        for i, patches in enumerate(loader):           # (1, 36, C, 128, 128, 64)
            tokens = model(patches.to(device), grid_size=GRID)      # (1, T, F)
            feats.append(tokens.mean(dim=1).float().cpu().numpy())  # the head's mean over tokens
            if (i + 1) % 50 == 0 or i + 1 == len(df):
                print(f"   {i + 1}/{len(df)}  ({time.time() - t0:.0f}s)", flush=True)
    F = np.concatenate(feats)
    if len(F) != len(df) or not np.isfinite(F).all():
        raise RuntimeError(f"{cohort}: bad features ({len(F)} rows for {len(df)} scans, or non-finite)")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, keys=df["key"].values, features=F.astype(np.float32), preproc=preproc,
                        model="spectre-large-pretrained", grid=np.array(GRID))
    print(f"   saved {F.shape} -> {out}  ({time.time() - t0:.0f}s)")
    return out


def load_spectre_features(cohort: str, preproc: str) -> dict:
    """{patient key: SPECTRE feature vector}, checking the preprocessing it was made with."""
    path = SPECTRE_FEATURE_DIRS[preproc] / f"features_{cohort.lower()}.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- run: python extract_spectre_features.py --preproc {preproc}")
    d = np.load(path, allow_pickle=True)
    if str(d["preproc"]) != preproc:
        raise ValueError(f"{path} holds '{d['preproc']}' features, but '{preproc}' was requested")
    return dict(zip(d["keys"], d["features"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--preproc", required=True, choices=list(FEATURE_DIRS),
                    help="hu = corrected (use this); published = the submitted, squeezed input")
    ap.add_argument("--cohorts", nargs="+", default=list(COHORTS))
    ap.add_argument("--workers", type=int, default=8, help="CPU processes preparing scans")
    ap.add_argument("--force", action="store_true", help="overwrite existing features")
    a = ap.parse_args()
    for c in a.cohorts:
        extract(c, a.preproc, a.workers, a.force)
