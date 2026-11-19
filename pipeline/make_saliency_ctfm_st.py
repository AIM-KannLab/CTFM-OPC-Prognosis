"""Where in the scan does CT-FM-ST's risk score come from? Exact contribution maps for the final model.

CT-FM-ST is a linear Cox model on the frozen CT-FM encoder's last feature map, averaged over space:
    risk = sum_c beta_c * (mean_v A_c(v) - mu_c) / sigma_c
so the score is exactly the spatial mean of a per-location map plus a constant:
    map(v) = sum_c (beta_c / sigma_c) * A_c(v)        (a class activation map, CAM)
Positive = that location pushes the patient's risk up, negative = down. Nothing is approximated
(no gradients); the only limitation is resolution: the last feature map is coarse (each cell spans
about 16 x 16 mm in-plane), and it is upsampled for display.

The model is the final run's (outputs/20261019_072229_main) Radcure-trained CT-FM-ST, the same one the
external validation applies to Maastro. Each scan goes through exactly the feature preprocessing of
that run (extract_features.build_transform('hu')); the rebuilt features are checked against the
saved ones before a map is drawn. Outlines: automatic segmentations (config.yaml data.radcure_masks /
data.maastro_masks; label 1 = primary tumour, label 2 = involved nodes), on the same grid as the CT.

Patients shown: from the Radcure held-out TEST split for the endpoint, and from Maastro (all external),
the 2 highest- and 2 lowest-risk patients with a primary-tumour mask.

    CUDA_VISIBLE_DEVICES=0 python3 make_saliency_ctfm_st.py --endpoint os
-> <outputs.dir>/figures/saliency_ctfm_st_<endpoint>_{radcure,maastro}.png  (+ .csv with the in-mask shares)
"""
from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path

os.environ.setdefault("A1_DATA", "corrected")
os.environ.setdefault("A1_FEATURES", "hu")
os.environ.setdefault("A1_NSTAGE", "ajcc8")
os.environ.setdefault("A1_IMAGES", "original")

import matplotlib                                    # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
import nibabel as nib                                # noqa: E402
import numpy as np                                   # noqa: E402
import pandas as pd                                  # noqa: E402
import torch                                         # noqa: E402
import torch.nn.functional as F                      # noqa: E402
from matplotlib.lines import Line2D                  # noqa: E402
from monai.transforms import (Compose, CropForegroundd, EnsureTyped, Lambdad, LoadImaged,  # noqa: E402
                              Orientationd, ScaleIntensityRanged)

from cohort import build_cohort, split_dataset_ratio          # noqa: E402
from extract_features import load_features, to_hu             # noqa: E402
from make_report import INK, INK2, SURFACE                    # noqa: E402  (applies the shared style)
from ctfm_config import ctfm_encoder, data_path, outputs_dir  # noqa: E402

HERE = Path(__file__).resolve().parent
RUN = outputs_dir() / "20261019_072229_main/results_corrected_hu_n8_20261019_072229"
MASKS = {"Radcure": data_path("data.radcure_masks"), "Maastro": data_path("data.maastro_masks")}
NAME = {"os": "OS", "efs": "RFS", "lr": "LR", "dr": "DM"}
PRIMARY, NODE = "#ffd60a", "#00e5ff"          # outline colours, apart from the red/blue of the map


def mask_path(cohort, key):                    # TSR_<id>_0000.nii.gz -> TSR_<id>.nii.gz (TS_ for Maastro)
    return MASKS[cohort] / key.replace("_0000.nii.gz", ".nii.gz")


def transform():
    """The 'hu' feature preprocessing of extract_features.py, applied to the scan and its mask together."""
    return Compose([
        LoadImaged(keys=["image", "label"], ensure_channel_first=True), EnsureTyped(keys=["image", "label"]),
        Lambdad(keys="image", func=to_hu),
        Orientationd(keys=["image", "label"], axcodes="SPL"),
        ScaleIntensityRanged(keys="image", a_min=-1024, a_max=2048, b_min=0, b_max=1, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image", allow_smaller=True),
    ])


LABEL = {"ctfm_st": "CT-FM-ST", "ctfm_lp": "CT-FM-LP"}
BASE = outputs_dir() / "20261019_072229_main/results_baselines_corrected_hu_n8_20261019_072229"


def get_model(model, endpoint):
    """(risk(raw features) -> score, per-channel weight on the raw feature map). Both models are linear
    in the pooled features, so map(v) = sum_c weight_c * A_c(v) and score = mean_v map(v) + constant."""
    if model == "ctfm_st":
        m = pickle.load(open(RUN / "ctfm_st" / "models" / f"Radcure_{endpoint}_primary_fullshot.pkl", "rb"))
        beta = m["model"].params_.values
        mu, sd = np.asarray(m["preprocess"]["scaler_mean"]), np.asarray(m["preprocess"]["scaler_scale"])
        return (lambda X: ((X - mu) / sd) @ beta), beta / sd
    seeds = [np.load(BASE / "ctfm_lp" / "models" / f"Radcure_{endpoint}_fullshot_seed{s}.npz") for s in range(5)]
    risk = lambda X: np.mean([((X - z["scaler_mean"]) / z["scaler_scale"]) @ z["weight"] for z in seeds], axis=0)
    return risk, np.mean([z["weight"] / z["scaler_scale"] for z in seeds], axis=0)       # mean of the 5 seeds


def pick(cohort, endpoint, risk_fn, n_each):
    d = build_cohort(cohort, endpoint)
    pool = split_dataset_ratio(d)[2] if cohort == "Radcure" else d          # Radcure: held-out test split
    feats = load_features(cohort)
    pool = pool.assign(risk=risk_fn(np.stack([feats[k] for k in pool.key])))
    pool["pct"] = pool.risk.rank(pct=True) * 100                  # percentile within the whole set
    n_all = len(pool)
    pool = pool[[mask_path(cohort, k).exists() for k in pool.key]].copy()
    pool["has_primary"] = [bool((np.asarray(nib.load(mask_path(cohort, k)).dataobj) == 1).any()) for k in pool.key]
    pool = pool[pool.has_primary].sort_values("risk")
    return pd.concat([pool.tail(n_each).iloc[::-1], pool.head(n_each)]), n_all


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--endpoint", default="os", choices=list(NAME))
    ap.add_argument("--model", default="ctfm_st", choices=list(LABEL),
                    help="ctfm_st (ridge Cox) or ctfm_lp (linear layer, mean of its 5 seeds)")
    ap.add_argument("--cohorts", nargs="+", default=["Radcure", "Maastro"])
    ap.add_argument("--n", type=int, default=2, help="highest- and lowest-risk patients per cohort")
    ap.add_argument("--out", default=str(outputs_dir() / "figures"))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    risk_fn, weight = get_model(a.model, a.endpoint)
    w = torch.tensor(weight, dtype=torch.float32, device=dev)                         # per-channel weight
    if a.model != "ctfm_st":                     # how close is it to CT-FM-ST? same features, other weights
        st_fn, st_w = get_model("ctfm_st", a.endpoint)
        t = split_dataset_ratio(build_cohort("Radcure", a.endpoint))[2]
        f = load_features("Radcure"); X = np.stack([f[k] for k in t.key])
        rho = pd.Series(risk_fn(X)).corr(pd.Series(st_fn(X)), method="spearman")
        cos = float(weight @ st_w / np.linalg.norm(weight) / np.linalg.norm(st_w))
        print(f"{LABEL[a.model]} vs CT-FM-ST: weight cosine similarity {cos:.2f}; risk rank correlation on the Radcure test set {rho:.2f}")

    enc = ctfm_encoder().to(dev).eval()
    tf = transform()
    rows = []
    for cohort in a.cohorts:
        chosen, n_pool = pick(cohort, a.endpoint, risk_fn, a.n)
        saved = load_features(cohort)
        maps = []
        for r in chosen.itertuples():
            item = tf({"image": str(r.path), "label": str(mask_path(cohort, r.key))})
            x = item["image"].unsqueeze(0).to(dev)
            with torch.no_grad():
                A = enc(x)[-1][0]                                                   # (512, d, h, w)
            pooled = A.mean(dim=(1, 2, 3)).cpu().numpy()
            if not np.allclose(pooled, saved[r.key], atol=1e-3, rtol=1e-3):
                raise SystemExit(f"{r.key}: rebuilt features differ from the saved ones -- not the run's model input")
            cam = torch.einsum("c,cdhw->dhw", w, A)
            cam_up = F.interpolate(cam[None, None], size=x.shape[2:], mode="trilinear", align_corners=False)[0, 0].cpu().numpy()
            img, lab = item["image"][0].numpy(), item["label"][0].numpy().round().astype(int)
            body = img > 0
            share = lambda region, v: float(np.clip(v, 0, None)[region].sum() / max(np.clip(v, 0, None)[body].sum(), 1e-12))
            absshare = lambda region: float(np.abs(cam_up)[region].sum() / max(np.abs(cam_up)[body].sum(), 1e-12))
            tum = (lab == 1) | (lab == 2)
            rows.append({"cohort": cohort, "key": r.key, "risk_percentile": round(r.pct, 1), "risk": r.risk,
                         "event": int(r.event), "time_days": float(r.time),
                         "tumour_nodes_share_of_body_volume": float(tum[body].mean()),
                         "share_of_|map|_in_tumour_nodes": absshare(tum),
                         "share_of_positive_map_in_tumour_nodes": share(tum, cam_up),
                         "feature_map_size": tuple(A.shape[1:])})
            maps.append((r, img, lab, cam_up))
        draw(maps, cohort, a.endpoint, out / f"saliency_{a.model}_{a.endpoint}_{cohort.lower()}.png", n_pool, LABEL[a.model])
    pd.DataFrame(rows).to_csv(out / f"saliency_{a.model}_{a.endpoint}.csv", index=False)
    print(pd.DataFrame(rows)[["cohort", "risk_percentile", "event", "tumour_nodes_share_of_body_volume",
                              "share_of_|map|_in_tumour_nodes", "share_of_positive_map_in_tumour_nodes",
                              "feature_map_size"]].to_string(index=False))
    print(f"wrote {out}/saliency_{a.model}_{a.endpoint}_*.png")


def draw(maps, cohort, endpoint, path, n_pool, label="CT-FM-ST"):
    vmax = np.percentile(np.concatenate([np.abs(c[3]).ravel() for c in maps]), 99.5)       # one scale for all rows
    fig, axes = plt.subplots(len(maps), 3, figsize=(10.5, 3.55 * len(maps)))
    for i, (r, img, lab, cam) in enumerate(maps):
        prim_area = (lab == 1).sum(axis=(1, 2)); node_area = (lab == 2).sum(axis=(1, 2))
        s0 = int(prim_area.argmax())
        s2 = int(node_area.argmax()) if node_area.max() > 0 and abs(int(node_area.argmax()) - s0) > 2 else min(s0 + 6, img.shape[0] - 1)
        slices = sorted({max(s0 - 6, 0), s0, s2})
        while len(slices) < 3:
            slices.append(min(slices[-1] + 3, img.shape[0] - 1))
        hu = img * 3072 - 1024
        for j, s in enumerate(slices[:3]):
            ax = axes[i, j]
            ax.imshow(hu[s], cmap="gray", vmin=-200, vmax=275, interpolation="nearest")
            c = cam[s] / vmax
            rgba = plt.cm.RdBu_r(np.clip(c * 0.5 + 0.5, 0, 1))
            rgba[..., 3] = np.clip(np.abs(c), 0, 1) * 0.7
            ax.imshow(rgba, interpolation="bilinear")
            for lbl, col in ((1, PRIMARY), (2, NODE)):
                if (lab[s] == lbl).any():
                    ax.contour(lab[s] == lbl, levels=[0.5], colors=[col], linewidths=1.4)
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            ax.set_title(f"slice {s}" + ("  (largest tumour area)" if s == s0 else ""), fontsize=8.5, color=INK2, loc="left")
        outcome = (f"died at {r.time / 30.44:.0f} mo" if endpoint == "os" and r.event else
                   f"event at {r.time / 30.44:.0f} mo" if r.event else f"no event, followed {r.time / 30.44:.0f} mo")
        axes[i, 0].set_ylabel(f"{'HIGH' if r.pct >= 50 else 'LOW'} risk\n{r.pct:.0f}th percentile\n{outcome}",
                              fontsize=9, color=INK, rotation=0, ha="right", va="center", labelpad=12)
    where = "held-out test set" if cohort == "Radcure" else "external cohort"
    fig.suptitle(f"{label} ({NAME[endpoint]}): where the risk score comes from — {cohort}, {where} "
                 f"(n={n_pool}), 2 highest- and 2 lowest-risk patients", x=0.01, ha="left", fontsize=11, color=INK)
    handles = [Line2D([], [], color=PRIMARY, lw=2, label="primary tumour (auto-segmented)"),
               Line2D([], [], color=NODE, lw=2, label="involved nodes (auto-segmented)"),
               Line2D([], [], color=plt.cm.RdBu_r(0.95), lw=6, label="raises the risk score"),
               Line2D([], [], color=plt.cm.RdBu_r(0.05), lw=6, label="lowers the risk score")]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8.5, bbox_to_anchor=(0.5, -0.005))
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()
