"""Where does CT-FM's OS signal come from? Cohort-level contribution maps and a confounding check.

Part 1 (GPU) -- every out-of-sample patient: the Radcure held-out OS test split and all of Maastro.
  The exact contribution map of make_saliency_ctfm_st.py (CT-FM-ST and CT-FM-LP). Per patient: the
  share of the map's absolute contribution that falls inside primary tumour + involved nodes
  (automatic segmentations), against the share of the body volume they occupy, and the share that
  falls in air (outside the body). Enrichment = map share / volume share; 1 = no more than its size.
  The rebuilt features are checked against the run's saved features for every patient.

Part 2 (CPU) -- the same patients: tumour and nodal volume (from the masks), neck size (body volume
  and mean axial body area in the scan block) and body composition (fat and muscle fractions of the
  body voxels, HU -190..-30 and -29..150) from the CT. Spearman correlation of each with the CT-FM-ST
  score, and Cox models for OS: clinical score + CT-FM score, then adding tumour/nodal volume, then
  neck size and composition. Scores are the run's saved out-of-sample predictions.

    CUDA_VISIBLE_DEVICES=0 python3 explainability_os.py
-> <outputs.dir>/explainability_os/{saliency_per_patient.csv, confounding_per_patient.csv, summary.md}
(In this repository the module is used by explainability_maps.py, which re-points RUN / EXT / OUT.)
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("A1_DATA", "corrected")
os.environ.setdefault("A1_FEATURES", "hu")
os.environ.setdefault("A1_NSTAGE", "ajcc8")
os.environ.setdefault("A1_IMAGES", "original")

import nibabel as nib                                # noqa: E402
import numpy as np                                   # noqa: E402
import pandas as pd                                  # noqa: E402
import torch                                         # noqa: E402
import torch.nn.functional as F                      # noqa: E402
from lifelines import CoxPHFitter                    # noqa: E402
from scipy.stats import spearmanr, wilcoxon          # noqa: E402

from cohort import build_cohort, split_dataset_ratio                       # noqa: E402
from extract_features import load_features                                 # noqa: E402
from make_saliency_ctfm_st import RUN, get_model, mask_path, transform     # noqa: E402
from ctfm_config import ctfm_encoder, outputs_dir                          # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = outputs_dir() / "explainability_os"
EXT = outputs_dir() / "20261019_072229_main/results_external_corrected_hu_n8_20261019_072229/predictions"
VOXEL_ML = 1 * 1 * 3 / 1000                                   # stored grid 1 x 1 x 3 mm


def patients():
    rad = build_cohort("Radcure", "os")
    rad = split_dataset_ratio(rad)[2].assign(cohort="Radcure")                  # held-out test split
    maa = build_cohort("Maastro", "os").assign(cohort="Maastro")                # external, all
    d = pd.concat([rad, maa], ignore_index=True)
    d["mask"] = [mask_path(c, k) for c, k in zip(d.cohort, d.key)]
    return d


def part1(d: pd.DataFrame) -> pd.DataFrame:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc = ctfm_encoder().to(dev).eval()
    W = {m: torch.tensor(get_model(m, "os")[1], dtype=torch.float32, device=dev) for m in ("ctfm_st", "ctfm_lp")}
    saved = {c: load_features(c) for c in ("Radcure", "Maastro")}
    tf, rows = transform(), []
    for i, r in enumerate(d.itertuples()):
        if not r.mask.exists():
            continue
        item = tf({"image": str(r.path), "label": str(r.mask)})
        x = item["image"].unsqueeze(0).to(dev)
        with torch.no_grad():
            A = enc(x)[-1][0]
        if not np.allclose(A.mean(dim=(1, 2, 3)).cpu().numpy(), saved[r.cohort][r.key], atol=1e-3, rtol=1e-3):
            raise SystemExit(f"{r.key}: rebuilt features differ from the saved ones")
        img, lab = item["image"][0].numpy(), item["label"][0].numpy().round().astype(int)
        body, tum = img > 0, (lab == 1) | (lab == 2)
        if not tum.any():
            continue
        rec = {"cohort": r.cohort, "key": r.key, "volume_share_tumour_nodes": tum[body].mean()}
        for m, w in W.items():
            cam = torch.einsum("c,cdhw->dhw", w, A)
            cam = F.interpolate(cam[None, None], size=x.shape[2:], mode="trilinear", align_corners=False)[0, 0].cpu().numpy()
            a, pos = np.abs(cam), np.clip(cam, 0, None)
            rec[f"{m}_map_share_tumour_nodes"] = a[tum].sum() / a[body].sum()
            rec[f"{m}_pos_share_tumour_nodes"] = pos[tum].sum() / max(pos[body].sum(), 1e-12)
            rec[f"{m}_map_share_air"] = a[~body].sum() / a.sum()
            rec[f"{m}_mean_in_tumour_minus_rest"] = cam[tum].mean() - cam[body & ~tum].mean()
        rows.append(rec)
        if (i + 1) % 50 == 0:
            print(f"  part 1: {i + 1}/{len(d)}", flush=True)
    s = pd.DataFrame(rows)
    for m in ("ctfm_st", "ctfm_lp"):
        s[f"{m}_enrichment"] = s[f"{m}_map_share_tumour_nodes"] / s.volume_share_tumour_nodes
    return s


def part2(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in d.itertuples():
        img = np.asarray(nib.load(str(r.path)).dataobj, dtype=np.float32)
        body = img > 0
        hu = 475 * img - 200
        rec = {"cohort": r.cohort, "key": r.key, "time": r.time, "event": r.event, "age": r.age,
               "T_Stage": r.T_Stage, "N_Stage": r.N_Stage, "HPV": r.HPV,
               "body_volume_ml": body.sum() * VOXEL_ML,
               "body_area_cm2": body.sum(axis=(0, 1))[body.any(axis=(0, 1))].mean() / 100,
               "fat_fraction": ((hu >= -190) & (hu <= -30) & body).sum() / body.sum(),
               "muscle_fraction": ((hu >= -29) & (hu <= 150) & body).sum() / body.sum()}
        if r.mask.exists():
            lab = np.asarray(nib.load(str(r.mask)).dataobj)
            rec["tumour_ml"] = (lab == 1).sum() * VOXEL_ML
            rec["nodes_ml"] = (lab == 2).sum() * VOXEL_ML
        rows.append(rec)
    c = pd.DataFrame(rows)
    for cohort, src in (("Radcure", RUN / "{arm}/predictions/Radcure_os_primary_fullshot.npz"),
                        ("Maastro", EXT / "Radcure_to_Maastro_os_{arm}_refit_train_primary.npz")):
        for arm, col in (("ctfm_st", "ctfm_score"), ("clinical_only", "clinical_score")):
            z = np.load(str(src).format(arm=arm), allow_pickle=True)
            m = dict(zip(z["keys"], z["risk"]))
            idx = c.cohort == cohort
            c.loc[idx, col] = c.loc[idx, "key"].map(m)
    return c


def cox(df, covs):
    x = df[["time", "event"] + covs].dropna().copy()
    for v in covs:
        x[v] = (x[v] - x[v].mean()) / x[v].std()
    f = CoxPHFitter().fit(x, "time", "event")
    s = f.summary
    return {v: (s.loc[v, "exp(coef)"], s.loc[v, "exp(coef) lower 95%"], s.loc[v, "exp(coef) upper 95%"], s.loc[v, "p"]) for v in covs}, len(x)


def summarise(s: pd.DataFrame, c: pd.DataFrame) -> str:
    q = lambda v: f"{np.median(v):.3f} [{np.percentile(v, 25):.3f}–{np.percentile(v, 75):.3f}]"
    L = ["# Where does CT-FM's OS signal come from?", "",
         "## 1. Contribution maps, every out-of-sample patient", "",
         "Median [IQR] over patients. Enrichment = share of the map's absolute contribution inside tumour + "
         "nodes ÷ their share of the body volume (1 = no more than their size). Wilcoxon: map share vs volume share.", "",
         "| Cohort | Model | n | Volume share, tumour + nodes | Map share, tumour + nodes | Enrichment | Patients with enrichment > 1 | Wilcoxon p | Map share in air |",
         "|---|---|---|---|---|---|---|---|---|"]
    for co in ("Radcure", "Maastro"):
        g = s[s.cohort == co]
        for m, lab in (("ctfm_st", "CT-FM-ST"), ("ctfm_lp", "CT-FM-LP")):
            p = wilcoxon(g[f"{m}_map_share_tumour_nodes"], g.volume_share_tumour_nodes).pvalue
            L.append(f"| {co} | {lab} | {len(g)} | {q(g.volume_share_tumour_nodes)} | {q(g[f'{m}_map_share_tumour_nodes'])} | "
                     f"{q(g[f'{m}_enrichment'])} | {(g[f'{m}_enrichment'] > 1).mean():.0%} | {p:.2g} | {q(g[f'{m}_map_share_air'])} |")
    L += ["", "## 2. What does the CT-FM-ST score track? (Spearman ρ, out-of-sample patients)", "",
          "| Variable | Radcure test ρ (p) | Maastro ρ (p) |", "|---|---|---|"]
    for v, lab in (("tumour_ml", "primary tumour volume"), ("nodes_ml", "nodal volume"),
                   ("body_volume_ml", "body volume in the scan block"), ("body_area_cm2", "mean axial body area (neck size)"),
                   ("fat_fraction", "fat fraction"), ("muscle_fraction", "muscle fraction"), ("age", "age"),
                   ("T_Stage", "T stage"), ("N_Stage", "N stage"), ("HPV", "HPV positive"), ("clinical_score", "clinical risk score")):
        cells = []
        for co in ("Radcure", "Maastro"):
            g = c[c.cohort == co][["ctfm_score", v]].dropna()
            rho, p = spearmanr(g.ctfm_score, g[v])
            cells.append(f"{rho:+.2f} ({p:.2g})")
        L.append(f"| {lab} | {cells[0]} | {cells[1]} |")
    L += ["", "## 3. Does the CT-FM-ST score stay prognostic after adjustment? (Cox, OS, HR per SD)", ""]
    c = c.assign(log_tumour=np.log1p(c.tumour_ml), log_nodes=np.log1p(c.nodes_ml))
    models = [("clinical + CT-FM", ["clinical_score", "ctfm_score"]),
              ("+ tumour and nodal volume", ["clinical_score", "ctfm_score", "log_tumour", "log_nodes"]),
              ("+ neck size and composition", ["clinical_score", "ctfm_score", "body_area_cm2", "fat_fraction", "muscle_fraction"]),
              ("+ both", ["clinical_score", "ctfm_score", "log_tumour", "log_nodes", "body_area_cm2", "fat_fraction", "muscle_fraction"])]
    L += ["| Cohort | Model | n (events) | CT-FM score HR (95% CI), p | Other terms, HR (p) |", "|---|---|---|---|---|"]
    for co in ("Radcure", "Maastro"):
        g = c[c.cohort == co]
        for name, covs in models:
            res, n = cox(g, covs)
            hr = res["ctfm_score"]
            others = ", ".join(f"{k.replace('_', ' ')} {v[0]:.2f} ({v[3]:.2g})" for k, v in res.items() if k not in ("ctfm_score", "clinical_score"))
            L.append(f"| {co} | {name} | {n} ({int(g.dropna(subset=covs).event.sum())}) | "
                     f"{hr[0]:.2f} ({hr[1]:.2f}–{hr[2]:.2f}), p={hr[3]:.2g} | {others or '—'} |")
    L += ["", "Tumour/nodal volumes are from automatic segmentations; neck size and composition are measured inside the "
          "stored 160×160×64 scan block (HU window −200…275). Scores are the final run's out-of-sample predictions "
          "(Radcure test split; Maastro external)."]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    d = patients()
    print(f"patients: Radcure test {int((d.cohort == 'Radcure').sum())}, Maastro {int((d.cohort == 'Maastro').sum())}")
    c = part2(d); c.to_csv(OUT / "confounding_per_patient.csv", index=False)
    print("part 2 done", flush=True)
    s = part1(d); s.to_csv(OUT / "saliency_per_patient.csv", index=False)
    text = summarise(s, c)
    (OUT / "summary.md").write_text(text)
    print(text)
