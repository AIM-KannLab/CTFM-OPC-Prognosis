"""Locate and read the published full-shot CT-FM-ST inference bundles.

Each bundle is a state_dict holding the frozen backbone plus three things we
actually need:
    scaler_mean   (512,)    feature standardisation fitted on that cohort's train split
    scaler_scale  (512,)
    linear_head.weight (1,512)   the Cox coefficients

    risk = w . ((feature - scaler_mean) / scaler_scale)

WARNING -- the OS bundles are corrupt. `few_shot_single_institution_running.py:786`
refits `StandardScaler` on features that were already standardised at line 734, so
the persisted scaler is the identity (mean ~ 1e-8, scale == 1) while the Cox
coefficients were fitted on z-scored features. Applying that bundle feeds raw
pooled features to coefficients expecting standardised ones. The efs / lr / dr
scripts have no such line and their bundles are sound. `load_bundle` refuses OS
unless explicitly overridden.
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import torch

from ctfm_config import data_path

# The submitted study's code + saved models ("working_space"): config.yaml submitted.workspace (optional).
WS = data_path("submitted.workspace")
BUNDLE_DIR = {
    "os":  WS / "results_FewShot_single_institution",
    "efs": WS / "results_FewShot_single_institution_efs",
    "lr":  WS / "results_FewShot_single_institution_lr",
    "dr":  WS / "results_FewShot_single_institution_dr",
}
CORRUPT_SCALER = {"os"}


def bundle_path(endpoint: str, cohort: str) -> Path:
    """Newest final_models_* bundle for this endpoint and cohort."""
    pat = str(BUNDLE_DIR[endpoint] / f"final_models_{endpoint}_*" / cohort /
              "inference_fulltrain_segresnet_pretrained_frozen.pt")
    hits = sorted(glob.glob(pat))
    if not hits:
        raise FileNotFoundError(f"no bundle for {endpoint}/{cohort} matching {pat}")
    return Path(hits[-1])


def load_bundle(endpoint: str, cohort: str, allow_corrupt: bool = False):
    """Return (scaler_mean, scaler_scale, coef) as float64 numpy arrays."""
    if endpoint in CORRUPT_SCALER and not allow_corrupt:
        raise ValueError(
            f"the {endpoint} bundle has an identity scaler (known bug at "
            f"few_shot_single_institution_running.py:786) and cannot be used for "
            f"inference. Use --source refit, or pass allow_corrupt=True to inspect it.")
    sd = torch.load(bundle_path(endpoint, cohort), map_location="cpu", weights_only=False)
    mean = sd["scaler_mean"].numpy().astype(np.float64)
    scale = sd["scaler_scale"].numpy().astype(np.float64)
    coef = sd["linear_head.weight"].numpy().astype(np.float64).ravel()
    return mean, scale, coef


def bundle_risk(features: np.ndarray, endpoint: str, cohort: str,
                allow_corrupt: bool = False) -> np.ndarray:
    """Risk score (higher = worse) for `features` under a published bundle."""
    mean, scale, coef = load_bundle(endpoint, cohort, allow_corrupt)
    return ((features - mean) / scale) @ coef


def reported_cindex(endpoint: str, cohort: str):
    """The internal-test C-index the bundle was saved with, if recorded."""
    p = bundle_path(endpoint, cohort).parent / "best_cindex_segresnet_pretrained_frozen.txt"
    try:
        return float(p.read_text().strip())
    except Exception:
        return None
