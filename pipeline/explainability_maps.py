"""Explainability / failure analysis (Reviewer 3, comment 6) on the CT-FM-ST model of the main run.

Run: outputs.paper_run in config.yaml, or the newest <outputs.dir>/<run>_study (published: 20261104_054739).

Same analysis as explainability_os.py + make_saliency_ctfm_st.py (which used the corrected-pipeline run 20261019_072229),
re-pointed at the model whose numbers are in the revised paper:
  model    main run, CT-FM-ST, OS, Radcure full-shot (results_original_n8_<run>/ctfm_st/models)
  input    the run's "st" CT-FM input (runner.st_transform: SPL + ScaleIntensityRange(-200, 275) on the
           stored [0, 1] scans, 64 x 160 x 160); rebuilt features are checked against features_paper_st/ per patient
  scores   the run's saved out-of-sample predictions (Radcure OS test split; Maastro external, refit on Radcure train)
Part 1 (GPU): exact contribution map (CAM) per patient; share of |map| in tumour + nodes (automatic segmentations)
          vs their share of the body volume (enrichment), and share of the map in air. Body = stored scan > 0
          (the "st" input is squeezed into [0.421, 0.423], so the body cannot be read from it).
Part 2 (CPU): explainability_os.part2 unchanged (tumour/nodal volume, neck size, fat/muscle fraction, clinical score),
          Spearman correlations with the CT-FM-ST score, Cox models with adjustment.
Figures:  2 highest- and 2 lowest-risk patients per cohort (make_saliency_ctfm_st.draw), stored-scan display.
CT-FM-LP is not included (its input is a different, resampled volume); SPECTRE gradient x input not re-run.

    CUDA_VISIBLE_DEVICES=0 python3 explainability_maps.py [--limit N]   (--limit: smoke test on N patients)
-> <outputs.dir>/explainability_maps/{saliency_per_patient.csv, confounding_per_patient.csv, summary.md,
                                      saliency_ctfm_st_os_{radcure,maastro}.png, run_info.txt}
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ctfm_config import ctfm_encoder, outputs_dir, paper_run                    # noqa: E402
ROOT, RUN_ID = paper_run(must_exist=False)        # outputs.paper_run, or the newest run (checked in __main__)
import runner as PR                                                     # noqa: E402

PR.setup(data="original", features="st", root=ROOT)          # env + FEATURE_DIRS["published"] -> features_paper_st

import numpy as np                                                            # noqa: E402
import pandas as pd                                                           # noqa: E402
import torch                                                                  # noqa: E402
import torch.nn.functional as F                                               # noqa: E402
from monai.transforms import (Compose, CopyItemsd, CropForegroundd, EnsureTyped, LoadImaged,   # noqa: E402
                              Orientationd, ScaleIntensityRanged)
from scipy.stats import spearmanr, wilcoxon                                   # noqa: E402

import explainability_os as EO                                                # noqa: E402
import make_saliency_ctfm_st as MS                                            # noqa: E402
from extract_features import load_features                                    # noqa: E402

RUN = ROOT / f"results_original_n8_{RUN_ID}"
EXT = ROOT / f"results_external_original_n8_{RUN_ID}" / "predictions"
OUT = outputs_dir() / "explainability_maps"
MS.RUN, EO.RUN, EO.EXT = RUN, RUN, EXT


def st_transform_d():
    """runner.st_transform as a dictionary transform (scan + mask + the stored scan for the body mask)."""
    return Compose([LoadImaged(keys=["image", "label"], ensure_channel_first=True), EnsureTyped(keys=["image", "label"]),
                    Orientationd(keys=["image", "label"], axcodes="SPL"),
                    CopyItemsd(keys="image", names="raw"),
                    ScaleIntensityRanged(keys="image", a_min=-200, a_max=275, b_min=0, b_max=1, clip=True),
                    CropForegroundd(keys=["image", "label", "raw"], source_key="image", allow_smaller=True)])


def display_img(raw):
    """Stored [0, 1] scan -> the scale draw() expects (img * 3072 - 1024 = HU)."""
    hu = np.where(raw <= 0, -1024.0, 475.0 * raw - 200.0)
    return (hu + 1024.0) / 3072.0


def checks(risk_fn):
    """The model reproduces the run's saved out-of-sample OS predictions (Radcure test; Maastro external)."""
    out = []
    for cohort, f in (("Radcure", RUN / "ctfm_st/predictions/Radcure_os_primary_fullshot.npz"),
                      ("Maastro", EXT / "Radcure_to_Maastro_os_ctfm_st_refit_train_primary.npz")):
        z = np.load(f, allow_pickle=True)
        feats = load_features(cohort)
        mine = risk_fn(np.stack([feats[k] for k in z["keys"]]))
        rho = pd.Series(mine).corr(pd.Series(np.asarray(z["risk"], float)), method="spearman")
        out.append(f"{cohort}: n={len(mine)}, Spearman(model risk, saved risk) = {rho:.6f}")
        if rho < 0.999:
            raise SystemExit("model does not reproduce the saved predictions: " + out[-1])
    return out


def part1(d, w, dev, limit=None):
    enc = ctfm_encoder().to(dev).eval()
    saved = {c: load_features(c) for c in ("Radcure", "Maastro")}
    tf, rows, n = st_transform_d(), [], 0
    for r in d.itertuples():
        if not r.mask.exists():
            continue
        item = tf({"image": str(r.path), "label": str(r.mask)})
        x = item["image"].unsqueeze(0).to(dev)
        with torch.no_grad():
            A = enc(x)[-1][0]
        if not np.allclose(A.mean(dim=(1, 2, 3)).cpu().numpy(), saved[r.cohort][r.key], atol=1e-3, rtol=1e-3):
            raise SystemExit(f"{r.key}: rebuilt features differ from features_paper_st -- not the run's input")
        raw, lab = item["raw"][0].numpy(), item["label"][0].numpy().round().astype(int)
        body, tum = raw > 0, (lab == 1) | (lab == 2)
        if not tum.any():
            continue
        cam = torch.einsum("c,cdhw->dhw", w, A)
        cam = F.interpolate(cam[None, None], size=x.shape[2:], mode="trilinear", align_corners=False)[0, 0].cpu().numpy()
        a, pos = np.abs(cam), np.clip(cam, 0, None)
        rows.append({"cohort": r.cohort, "key": r.key, "volume_share_tumour_nodes": tum[body].mean(),
                     "ctfm_st_map_share_tumour_nodes": a[tum].sum() / a[body].sum(),
                     "ctfm_st_pos_share_tumour_nodes": pos[tum].sum() / max(pos[body].sum(), 1e-12),
                     "ctfm_st_map_share_air": a[~body].sum() / a.sum(),
                     "ctfm_st_mean_in_tumour_minus_rest": cam[tum].mean() - cam[body & ~tum].mean(),
                     "feature_map_size": tuple(A.shape[1:])})
        n += 1
        if n % 50 == 0:
            print(f"  part 1: {n} patients", flush=True)
        if limit and n >= limit:
            break
    s = pd.DataFrame(rows)
    s["ctfm_st_enrichment"] = s.ctfm_st_map_share_tumour_nodes / s.volume_share_tumour_nodes
    return s, enc


def figures(enc, w, risk_fn, dev):
    tf = st_transform_d()
    for cohort in ("Radcure", "Maastro"):
        chosen, n_pool = MS.pick(cohort, "os", risk_fn, 2)
        maps = []
        for r in chosen.itertuples():
            item = tf({"image": str(r.path), "label": str(MS.mask_path(cohort, r.key))})
            x = item["image"].unsqueeze(0).to(dev)
            with torch.no_grad():
                A = enc(x)[-1][0]
            cam = torch.einsum("c,cdhw->dhw", w, A)
            cam = F.interpolate(cam[None, None], size=x.shape[2:], mode="trilinear", align_corners=False)[0, 0].cpu().numpy()
            maps.append((r, display_img(item["raw"][0].numpy()), item["label"][0].numpy().round().astype(int), cam))
        MS.draw(maps, cohort, "os", OUT / f"saliency_ctfm_st_os_{cohort.lower()}.png", n_pool, "CT-FM-ST")


def summarise(s, c):
    q = lambda v: f"{np.median(v):.3f} [{np.percentile(v, 25):.3f}–{np.percentile(v, 75):.3f}]"
    L = [f"# Where does CT-FM-ST's OS signal come from? (main-run model, run {RUN_ID})", "",
         "## 1. Exact contribution maps, every out-of-sample patient with a tumour mask", "",
         "Median [IQR]. Enrichment = share of |map| inside tumour + nodes ÷ their share of the body volume (1 = no more "
         "than their size). Wilcoxon: map share vs volume share. Air = outside the body (stored scan = 0, ≤ −200 HU).", "",
         "| Cohort | n | Volume share, tumour + nodes | Map share, tumour + nodes | Enrichment | Patients with enrichment > 1 | Wilcoxon p | Map share in air |",
         "|---|---|---|---|---|---|---|---|"]
    for co in ("Radcure", "Maastro"):
        g = s[s.cohort == co]
        if len(g) < 2:
            continue
        p = wilcoxon(g.ctfm_st_map_share_tumour_nodes, g.volume_share_tumour_nodes).pvalue
        L.append(f"| {co} | {len(g)} | {q(g.volume_share_tumour_nodes)} | {q(g.ctfm_st_map_share_tumour_nodes)} | "
                 f"{q(g.ctfm_st_enrichment)} | {(g.ctfm_st_enrichment > 1).mean():.0%} | {p:.2g} | {q(g.ctfm_st_map_share_air)} |")
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
            res, n = EO.cox(g, covs)
            hr = res["ctfm_score"]
            others = ", ".join(f"{k.replace('_', ' ')} {v[0]:.2f} ({v[3]:.2g})" for k, v in res.items() if k not in ("ctfm_score", "clinical_score"))
            L.append(f"| {co} | {name} | {n} ({int(g.dropna(subset=covs).event.sum())}) | "
                     f"{hr[0]:.2f} ({hr[1]:.2f}–{hr[2]:.2f}), p={hr[3]:.2g} | {others or '—'} |")
    L += ["", "Tumour/nodal volumes from automatic segmentations; neck size and composition measured in the "
          "stored 160×160×64 scan block (HU window −200…275). Scores = the run's out-of-sample predictions "
          "(Radcure OS test split; Maastro external, model refit on the Radcure training set)."]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--limit", type=int, default=None, help="smoke test: stop part 1 after N patients, no figures")
    a = ap.parse_args()
    paper_run()                                   # stops with a clear message when there is no run to analyse
    OUT.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    risk_fn, weight = MS.get_model("ctfm_st", "os")
    info = checks(risk_fn)
    print("\n".join(info), flush=True)
    w = torch.tensor(weight, dtype=torch.float32, device=dev)
    d = EO.patients()
    print(f"patients: Radcure test {int((d.cohort == 'Radcure').sum())}, Maastro {int((d.cohort == 'Maastro').sum())}", flush=True)
    if a.limit:
        s, _ = part1(d, w, dev, a.limit)
        print(s.to_string()); raise SystemExit("smoke test OK")
    c = EO.part2(d); c.to_csv(OUT / "confounding_per_patient.csv", index=False)
    print("part 2 done", flush=True)
    s, enc = part1(d, w, dev)
    s.to_csv(OUT / "saliency_per_patient.csv", index=False)
    figures(enc, w, risk_fn, dev)
    text = summarise(s, c)
    (OUT / "summary.md").write_text(text)
    (OUT / "run_info.txt").write_text("\n".join(info) + f"\nmodel: {RUN}/ctfm_st/models/Radcure_os_primary_fullshot.pkl\n")
    print(text)
