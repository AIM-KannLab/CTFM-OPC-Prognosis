"""IBS (with 95% CI) of the submitted paper's own saved CT-FM-LP and SegRes-E2E networks, OS and RFS.

The submitted Table S6 / Figure 5 scored one saved network per cohort and endpoint
(inference_fulltrain_<type>.pt) with working_space/generate_calibration_pytorch_head.py. That script is
unchanged since it produced the submitted numbers (saved 2026-02-25 21:05, results 21:13), so this
wrapper imports it and calls its own functions -- data, split, transform, model loading, Breslow
baseline, IPCW Brier/IBS, calibration bins -- and adds only:
  * 95% CIs (2,000 event-stratified bootstrap resamples of the test patients, revision bootstrap_ci.py)
    for IBS and C-index;
  * a check that the re-run reproduces the submitted per-patient test risk scores (saved at the time);
  * an input check: every network saved the mean/SD of its training features (its scaler). The mean of
    the training features under a given input transform should reproduce it. Two inputs are tried:
      submitted  ScaleIntensityRange(-200, 275)   -- what the calibration script feeds every network
      hu_wide    ScaleIntensityRange(-1024, 2048) -- the OS networks' training input (verified for OS LP)
    Both are scored; the summary says which one each network was trained on.
OPTIONAL step of this repository: it needs the submitted study's code base and saved networks ("working_space":
generate_calibration_pytorch_head.py, results_FewShot_single_institution*_with_pytorch_head/final_models_*,
result_plot_main_FINALL/results_calibration_all_endpoints_2year, and the data/ folder that script reads), set as
submitted.workspace in config.yaml. Without them make_paper_style_figures.py uses the re-trained networks
(submitted.use_saved_segres_networks = auto).

Inference only (no training). Data: the submitted CSVs (OS, RFS unchanged by the revision's label fixes
for Maastro; Radcure RFS times were corrected later -- the submitted labels are used here on purpose,
to reproduce the submitted table).

    CUDA_VISIBLE_DEVICES=0 python3 optional/score_paper_networks_ibs.py [--out <outputs.dir>/paper_networks_ibs]
Writes <out>/summary.csv, <out>/bins.csv, <out>/<variant>_<cohort>_<ep>_<model>.npz (no patient IDs).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "pipeline"))       # the repository's pipeline modules
from ctfm_config import CFG, outputs_dir  # noqa: E402
OUT_DEFAULT = outputs_dir() / "paper_networks_ibs"
WS = CFG.path("submitted.workspace", required=True)     # the submitted study's code + saved models
if not (WS / "generate_calibration_pytorch_head.py").exists():
    raise SystemExit(f"submitted.workspace = {WS}: no generate_calibration_pytorch_head.py there -- this optional "
                     f"step needs the submitted study's code base and saved networks")
from bootstrap_ci import bootstrap_ci  # noqa: E402
from ipcw_brier import ibs_from_probs  # noqa: E402  (same formula as the submitted one, vectorised)

os.chdir(WS)                                   # the submitted script uses paths relative to working_space
sys.path.insert(0, str(WS))
import logging  # noqa: E402
import generate_calibration_pytorch_head as G  # noqa: E402
G.logger.setLevel(logging.WARNING)             # it logs every Brier computation
from monai.transforms import (Compose, CropForeground, EnsureType, LoadImage, Orientation,  # noqa: E402
                              ScaleIntensityRange, Spacing)
from lifelines import KaplanMeierFitter  # noqa: E402
from lifelines.utils import concordance_index  # noqa: E402

MODEL_DIRS = {   # generate_calibration_all_endpoints.py, the submitted run
    "os": WS / "results_FewShot_single_institution_with_pytorch_head/final_models_os_20251124_164923",
    "efs": WS / "results_FewShot_single_institution_efs_with_pytorch_head/final_models_efs_20260125_023022",
}
MODELS = {"segresnet_pretrained_frozen": "CT-FM-LP", "segresnet_random_finetuned": "SegRes-E2E"}
EP_DIR = {"os": "OS", "efs": "EFS"}
EP_NAME = {"os": "OS", "efs": "RFS"}
SUBMITTED = WS / "result_plot_main_FINALL/results_calibration_all_endpoints_2year"
VARIANTS = {"submitted": (-200, 275), "hu_wide": (-1024, 2048)}


def transform(a_min, a_max):
    return Compose([LoadImage(ensure_channel_first=True), EnsureType(), Orientation(axcodes="SPL"),
                    Spacing(pixdim=(1.0, 1.0, 3.0), mode="bilinear"),
                    ScaleIntensityRange(a_min=a_min, a_max=a_max, b_min=0, b_max=1, clip=True),
                    CropForeground(allow_smaller=True)])


class Paths(torch.utils.data.Dataset):
    def __init__(self, paths, tf):
        self.paths, self.tf = paths, tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return i, torch.as_tensor(np.asarray(self.tf(str(self.paths[i]))), dtype=torch.float32)


def load_images(paths, tf, workers):
    out = [None] * len(paths)
    dl = torch.utils.data.DataLoader(Paths(paths, tf), batch_size=None, shuffle=False, num_workers=workers)
    for i, x in dl:
        out[i] = x
    return out


@torch.no_grad()
def forward(model, imgs, device):
    feats, risk = [], []
    for x in imgs:
        f = model.backbone(x[None].float().to(device))
        f = f[-1] if isinstance(f, (list, tuple)) else f
        f = F.adaptive_avg_pool3d(f, 1).flatten(1) if f.dim() > 2 else f
        z = (f - model.scaler_mean) / (model.scaler_scale + 1e-12)
        feats.append(f.cpu().numpy()[0])
        risk.append(float(model.linear_head(z).cpu().numpy().ravel()[0]))
    return np.asarray(feats), np.asarray(risk)


def km_bins(surv730, t, e, n_bins=5):
    """generate_calibration_pytorch_head.create_calibration_plot's bins, plus each bin's KM 95% CI."""
    edges = np.unique(np.percentile(surv730, np.linspace(0, 100, n_bins + 1)))
    if len(edges) < 3:
        edges = np.linspace(surv730.min(), surv730.max(), n_bins + 1)
    rows = []
    for i in range(len(edges) - 1):
        last = i == len(edges) - 2
        m = (surv730 >= edges[i]) & ((surv730 <= edges[i + 1]) if last else (surv730 < edges[i + 1]))
        if m.sum() < 3:
            continue
        km = KaplanMeierFitter().fit(t[m], e[m])
        sf, ci = km.survival_function_, km.confidence_interval_survival_function_
        at = sf.index[sf.index <= 730]
        obs = km.survival_function_at_times(730).values[0] if 730 <= sf.index.max() else sf.iloc[-1, 0]
        lo, hi = (ci.loc[at[-1]].values if len(at) else (1.0, 1.0))
        rows.append({"bin": i, "predicted": float(surv730[m].mean()), "observed": float(obs),
                     "obs_lo": float(lo), "obs_hi": float(hi), "n": int(m.sum())})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    ap.add_argument("--cohorts", nargs="+", default=["Maastro", "Radcure"])
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    ap.add_argument("--models", nargs="+", default=["SegRes-E2E"], choices=list(MODELS.values()),
                    help="which saved networks to score (default: SegRes-E2E only)")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device {device} ({os.environ.get('CUDA_VISIBLE_DEVICES', 'all')})   out {out}", flush=True)
    cfg = G.Config()
    rows, bins = [], []
    for co in a.cohorts:
        build = G.build_radcure if co == "Radcure" else G.build_maastro
        splits = {}
        for ep in MODEL_DIRS:
            df = build(cfg, endpoint=ep)
            tr, va, te = G.split_dataset_ratio(df, cfg.train_frac, cfg.val_frac, cfg.test_frac, seed=42)
            splits[ep] = (tr, te)
        paths = sorted({str(p) for tr, te in splits.values() for p in list(tr.path) + list(te.path)})
        for var in a.variants:
            t0 = time.time()
            imgs = load_images(paths, transform(*VARIANTS[var]), a.workers)
            idx = {p: i for i, p in enumerate(paths)}
            print(f"[{co}/{var}] {len(paths)} scans loaded in {time.time() - t0:.0f}s", flush=True)
            for ep, (tr, te) in splits.items():
                for mtype, mname in MODELS.items():
                    if mname not in a.models:
                        continue
                    model = G.load_inference_model(G.find_model_file(MODEL_DIRS[ep], co, mtype), device)
                    f_tr, r_tr = forward(model, [imgs[idx[str(p)]] for p in tr.path], device)
                    f_te, r_te = forward(model, [imgs[idx[str(p)]] for p in te.path], device)
                    t_tr, e_tr = tr.time.values.astype(float), tr.event.values.astype(float)
                    t_te, e_te = te.time.values.astype(float), te.event.values.astype(float)
                    # input check: training-feature mean vs the scaler the network saved
                    mu, sd = model.scaler_mean.cpu().numpy(), model.scaler_scale.cpu().numpy() + 1e-12
                    zshift = float(np.mean(np.abs((f_tr.mean(0) - mu) / sd)))
                    sd_ratio = float(np.median(f_tr.std(0) / sd))
                    # submitted pipeline: Breslow on training risks, IPCW IBS over the clinical grid
                    bt, bs, centre = G.breslow_baseline_survival(t_tr, e_tr, r_tr)
                    tmax = bt[-1] if len(bt) else t_tr.max()
                    et = cfg.eval_times[cfg.eval_times <= tmax * 0.95]
                    surv = G.risk_to_survival_probs(r_te, bt, bs, et, centre)
                    # point estimate: the submitted function; bootstrap: the revision's identical, faster one
                    ibs_sub = G.compute_brier_score_ipcw(surv, et, t_te, e_te)[2]
                    ibs_fn = lambda i: ibs_from_probs(surv[i], t_te[i], e_te[i], et)
                    ibs, ibs_lo, ibs_hi, _ = bootstrap_ci(ibs_fn, t_te, e_te, B=a.boot)
                    assert abs(ibs - ibs_sub) < 1e-6, f"IBS functions disagree: {ibs} vs {ibs_sub}"
                    c, c_lo, c_hi, _ = bootstrap_ci(lambda i: concordance_index(t_te[i], -r_te[i], e_te[i]),
                                                    t_te, e_te, B=a.boot)
                    s730 = G.risk_to_survival_probs(r_te, bt, bs, np.array([730.0]), centre).flatten()
                    b = km_bins(s730, t_te, e_te)
                    ce = float(np.mean([abs(x["predicted"] - x["observed"]) for x in b])) if b else np.nan
                    # the submitted run's saved test risks and summary
                    sub_dir = SUBMITTED / EP_DIR[ep] / mname / co
                    z = np.load(sub_dir / f"raw_data_{mname}.npz", allow_pickle=True)
                    risk_match = float(np.max(np.abs(z["test_risk"] - r_te))) if len(z["test_risk"]) == len(r_te) else np.nan
                    s = pd.read_csv(sub_dir / f"summary_{mname}.csv").iloc[0]
                    row = {"variant": var, "cohort": co, "endpoint": EP_NAME[ep], "model": mname,
                           "n_test": len(te), "n_events": int(e_te.sum()),
                           "cindex": c, "cindex_lo": c_lo, "cindex_hi": c_hi,
                           "ibs": ibs, "ibs_lo": ibs_lo, "ibs_hi": ibs_hi, "calib_error": ce,
                           "submitted_cindex": float(s["C-index"]), "submitted_ibs": float(s["IBS"]),
                           "submitted_calib_error": float(s["CalibError"]),
                           "max_abs_diff_vs_submitted_test_risk": risk_match,
                           "train_feature_shift_sd": zshift, "train_feature_sd_ratio": sd_ratio}
                    rows.append(row)
                    bins += [{**{k: row[k] for k in ("variant", "cohort", "endpoint", "model")}, **x} for x in b]
                    np.savez(out / f"{var}_{co}_{ep}_{mname}.npz", test_times=t_te, test_events=e_te, test_risk=r_te,
                             surv=surv, eval_times=et, surv730=s730, train_risk=r_tr)
                    print(f"  {var:9} {co:7} {EP_NAME[ep]:3} {mname:10} C {c:.3f} (sub {row['submitted_cindex']:.3f})"
                          f"  IBS {ibs:.3f} [{ibs_lo:.3f}-{ibs_hi:.3f}] (sub {row['submitted_ibs']:.3f})"
                          f"  CE {ce:.3f} (sub {row['submitted_calib_error']:.3f})  risk diff {risk_match:.2g}"
                          f"  feat shift {zshift:.2f} SD, SD ratio {sd_ratio:.2f}", flush=True)
                    pd.DataFrame(rows).to_csv(out / "summary.csv", index=False)
                    pd.DataFrame(bins).to_csv(out / "bins.csv", index=False)
                    del model
                    torch.cuda.empty_cache()
            del imgs
    print(f"done -> {out / 'summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
