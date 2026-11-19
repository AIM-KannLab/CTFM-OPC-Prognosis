"""Re-run the SUBMITTED paper's pipeline with the revision's machinery -- to give the paper's numbers 95% CIs.

The submitted runs (Nov 2025 - Feb 2026, working_space/) cannot be re-run as they are: their scripts were edited
after submission (2026-03-28, 2026-09-13), and the models did not share one input. What each model actually got was
recovered from the paper's own saved models (their feature scalers; see memory/notes and verify below):

  CT-FM-ST    input "st": Orientation SPL + ScaleIntensityRange(-200, 275) on the stored [0,1] scans -> squeezed into
              [0.4211, 0.4232], 64 x 160 x 160 (all four endpoints, runs of 2026-01-30; match 0.0001 SD)
  CT-FM-LP,   input "lp": Orientation SPL + Spacing(1,1,3) (after SPL: resamples S and L -> 190 x 160 x 54) +
  SegRes-E2E  ScaleIntensityRange(-1024, 2048) on the stored scans -> [0.3333, 0.3337] (OS LP model, match 0.0003 SD)
  SPECTRE     extract_spectre_features.py --preproc published (squeezed [-1000, 1000] window); every endpoint
              here (the paper ran it for OS only)
  Cox loss    CT-FM-LP, SegRes-E2E, SPECTRE: the paper's CoxPHLoss, unflattened (B,1) x (B,) -> censored patients
              counted as events. CT-FM-ST (lifelines) is unaffected.
  Data        --data original (the paper's CSVs) or corrected. The launcher uses original for OS/RFS (reproduces the
              paper) and corrected for LR/DM (pure LR/DM events).
  SegRes      fp32, as the paper (no autocast).

Everything else is the revision code, run UNCHANGED: this wrapper only re-points a few module-level settings before
the target script starts (feature folder, scan cache, transform, seeds, loss), then runs it as __main__.

    python3 runner.py features --which st|lp           (GPU)  -> features_paper_{st,lp}/
    python3 runner.py verify --out <folder>            (CPU)  checks against the paper's saved models / CSVs
    python3 runner.py segres-cache                     (CPU)  -> volumes/paper_lp/
    python3 runner.py segres-worker --data original --endpoints os efs --gpu 2 --root <folder>   (GPU)
    python3 runner.py run --data original --features st --root <folder> -- run_a1.py --cohort Radcure ...
Launched by run_pipeline.sh.
"""
from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ctfm_config import CFG, cache_dir, ctfm_encoder, test_bootstrap_b, torch_device  # noqa: E402

CACHE = cache_dir()                           # config.yaml cache.dir: feature files and scan caches
FEAT = {"st": CACHE / "features_paper_st", "lp": CACHE / "features_paper_lp", "lp2": CACHE / "features_paper_lp2"}
# CT-FM-LP / SegRes-E2E input per endpoint, as recovered from the paper's saved LP models: the OS and DM runs
# (Nov 2025) used "lp", the RFS and LR runs (Jan 2026) "lp2" (the pytorch_head scripts as they are on disk now).
LP_INPUT = {"os": "lp", "dr": "lp", "efs": "lp2", "lr": "lp2"}
SPECTRE_FEAT = CACHE / "features_spectre_paper"
VOLUMES = {"lp": CACHE / "volumes" / "paper_lp", "lp2": CACHE / "volumes" / "paper_lp2"}
LP_SHAPE = (190, 160, 54)                     # input "lp" after SPL + Spacing(1,1,3), every scan
PAPER_CSV = {"original": "the paper's CSVs (new_*_data.csv)", "corrected": "new_*_data_corrected_main.csv"}


def st_transform():
    from monai.transforms import Compose, CropForeground, EnsureType, LoadImage, Orientation, ScaleIntensityRange
    return Compose([LoadImage(ensure_channel_first=True), EnsureType(), Orientation(axcodes="SPL"),
                    ScaleIntensityRange(a_min=-200, a_max=275, b_min=0, b_max=1, clip=True),
                    CropForeground(allow_smaller=True)])


def lp_transform():
    from monai.transforms import (Compose, CropForeground, EnsureType, LoadImage, Orientation,
                                  ScaleIntensityRange, Spacing)
    return Compose([LoadImage(ensure_channel_first=True), EnsureType(), Orientation(axcodes="SPL"),
                    Spacing(pixdim=(1.0, 1.0, 3.0), mode="bilinear"),
                    ScaleIntensityRange(a_min=-1024, a_max=2048, b_min=0, b_max=1, clip=True),
                    CropForeground(allow_smaller=True)])


def lp2_transform():
    from monai.transforms import (Compose, CropForeground, EnsureType, LoadImage, Orientation,
                                  ScaleIntensityRange, Spacing)
    return Compose([LoadImage(ensure_channel_first=True), EnsureType(), Orientation(axcodes="SPL"),
                    Spacing(pixdim=(1.0, 1.0, 3.0), mode="bilinear"),
                    ScaleIntensityRange(a_min=-200, a_max=275, b_min=0, b_max=1, clip=True),
                    CropForeground(allow_smaller=True)])


TRANSFORM = {"st": st_transform, "lp": lp_transform, "lp2": lp2_transform}


def paper_cox_loss(preds, times, events, eps: float = 1e-7):
    """CoxPHLoss of the submitted few_shot_single_institution_pytorch_head_*.py, verbatim (incl. its broadcast)."""
    import torch
    events = events.float().to(preds.device)
    order = torch.argsort(times, descending=True)
    preds = preds[order] - preds.max()
    events = events[order]
    log_risk = torch.log(torch.cumsum(torch.exp(preds), dim=0) + eps)
    cens = (preds - log_risk) * events
    n_events = events.sum()
    return -(cens.sum() / (n_events if n_events > 0 else 1.0))


paper_cox_loss.label = "the submitted study's CoxPHLoss, verbatim (runner.paper_cox_loss)"


def setup(data: str, features: str | None, root: Path | None, seeds: int = 5, segres_input: str = "lp"):
    """Point the revision modules at the paper's pipeline. Must run before the target script imports anything."""
    os.environ.update(A1_DATA=data, A1_FEATURES="published", A1_NSTAGE="ajcc8", A1_IMAGES="original",
                      A1_SEGRES_VARIANT="e2e", A1_SEGRES_PRECISION="fp32", A1_AUTOREPORT="0")
    if root is not None:
        os.environ["A1_OUTPUT_ROOT"] = str(root)
    sys.path.insert(0, str(HERE))
    import cohort
    if features:
        cohort.FEATURE_DIRS["published"] = FEAT[features]
    cohort.SPECTRE_FEATURE_DIRS["published"] = SPECTRE_FEAT
    cohort.VOLUME_DIR = VOLUMES[segres_input]
    import extract_features as EF
    _orig = EF.build_transform
    EF.build_transform = lambda p: TRANSFORM[segres_input]() if p == "published" else _orig(p)   # SegRes scans
    import subsample
    subsample.SEEDS[:] = list(range(seeds))
    import train_segres as TS                    # also gives run_baselines' linear head the paper loss
    TS.cox_loss = paper_cox_loss
    TS.VOLUME_DIR = VOLUMES[segres_input]
    TS.SPATIAL = LP_SHAPE
    _test_bootstrap()
    return cohort, TS


def _test_bootstrap():
    """Synthetic smoke test only (config testing.bootstrap_b): fewer bootstrap resamples in every
    bootstrap_ci / paired_bootstrap call. Unset (the default) = nothing changes."""
    B = test_bootstrap_b()
    if not B:
        return
    import bootstrap_ci as BC
    if getattr(BC, "_test_b", None) == B:
        return
    _ci, _pb = BC.bootstrap_ci, BC.paired_bootstrap
    BC.bootstrap_ci = lambda *a, **k: _ci(*a, **{**k, "B": B})
    BC.paired_bootstrap = lambda *a, **k: _pb(*a, **{**k, "B": B})
    BC._test_b = B
    print(f"!! TEST MODE (testing.bootstrap_b): {B} bootstrap resamples instead of the pipeline's", flush=True)


def cmd_features(a):
    """CT-FM features with the paper's input for every patient of both cohorts (encoder in eval mode)."""
    os.environ.update(A1_DATA="original", A1_FEATURES="published")
    sys.path.insert(0, str(HERE))
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    from cohort import COHORTS, build_cohort
    out = FEAT[a.which]
    out.mkdir(parents=True, exist_ok=True)
    tf = TRANSFORM[a.which]()

    class Scans(Dataset):
        def __init__(s, paths): s.p = list(paths)
        def __len__(s): return len(s.p)
        def __getitem__(s, i): return tf(str(s.p[i]))

    dev = torch_device()                        # runtime.device (auto: CUDA when available)
    enc = ctfm_encoder().to(dev).eval()         # model.ctfm_weights (project-lighter/ct_fm_feature_extractor)
    for c in COHORTS:
        path = out / f"features_{c.lower()}.npz"
        if path.exists() and not a.force:
            print(f"{path} exists -> skipped"); continue
        df = build_cohort(c, "os")
        feats = []
        with torch.no_grad():
            for i, x in enumerate(DataLoader(Scans(df["path"]), batch_size=None, num_workers=8)):
                feats.append(F.adaptive_avg_pool3d(enc(x[None].to(dev))[-1], 1).flatten().cpu().numpy())
                if (i + 1) % 200 == 0:
                    print(f"   {c} {i + 1}/{len(df)}", flush=True)
        X = np.stack(feats)
        if not np.isfinite(X).all():
            raise SystemExit(f"{c}: non-finite features")
        np.savez_compressed(path, keys=df["key"].values, features=X.astype(np.float32), preproc="published",
                            paper_input=a.which, transform=str(tf))
        print(f"saved {X.shape} -> {path}", flush=True)


def cmd_segres_cache(a):
    setup("original", None, None, segres_input=a.which)
    import train_segres as TS
    TS.cache_volumes(a.procs, a.force)


def cmd_segres_worker(a):
    setup(a.data, None, Path(a.root), segres_input=LP_INPUT[a.endpoints[0]])
    if len({LP_INPUT[e] for e in a.endpoints}) > 1:
        raise SystemExit("one worker = one input: give endpoints that share it (os+dr: lp, efs+lr: lp2)")
    import train_segres as TS
    sys.exit(TS.worker(("full", "few"), None, a.endpoints, a.gpu))


def cmd_run(a):
    setup(a.data, a.features, Path(a.root), seeds=a.seeds)
    script = HERE / a.script[0]
    sys.argv = [str(script)] + a.script[1:]
    print(f"[runner] data={a.data} input={a.features or '-'} seeds={a.seeds} root={a.root}\n"
          f"[runner] {' '.join(sys.argv)}", flush=True)
    if script.name == "run_baselines.py":         # SPECTRE on every endpoint (the paper: OS only)
        import run_baselines as RB
        from cohort import ENDPOINTS
        RB.MODEL_ENDPOINTS["spectre"] = list(ENDPOINTS)
        RB.main()
        return
    runpy.run_path(str(script), run_name="__main__")


def cmd_verify(a):
    """Gate: does this pipeline reproduce the paper? CT-FM-ST must match exactly; LP inputs are checked too."""
    import glob
    import numpy as np
    import pandas as pd
    import torch
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    WS = CFG.path("submitted.workspace")     # the submitted study's code + saved models (optional)
    if WS is None or not WS.is_dir():
        msg = (f"# Verification against the submitted study\n\nSkipped: submitted.workspace (the submitted study's saved models "
               f"and per-seed results) is not configured or does not exist ({WS}).\n")
        (out / "verify.md").write_text(msg)
        print(msg)
        sys.exit(0)
    setup("original", "st", None)
    import bundles
    from cohort import build_cohort, split_dataset_ratio
    from extract_features import load_features
    from bootstrap_ci import cindex
    from ipcw_brier import eval_times_for
    from models import fit_ctfm_st
    from subsample import stratified_subsample
    import cohort
    paper_csv = {"os": "results_FewShot_single_institution/fewshot_results_20260130_012109.csv",
                 "lr": "results_FewShot_single_institution_lr/fewshot_results_lr_20260130_023949.csv",
                 "dr": "results_FewShot_single_institution_dr/fewshot_results_dr_20260130_053021.csv",
                 "efs": "results_FewShot_single_institution_efs/fewshot_results_efs_20260130_040808.csv"}
    lp_models = {"os": "results_FewShot_single_institution_with_pytorch_head/final_models_os_20251124_164923",
                 "dr": "results_FewShot_single_institution_dr_with_pytorch_head/final_models_dr_20251127_155949",
                 "efs": "results_FewShot_single_institution_efs_with_pytorch_head/final_models_efs_20260125_023022",
                 "lr": "results_FewShot_single_institution_lr_with_pytorch_head/final_models_lr_20260130_023350"}
    L, bad = ["# Verification against the submitted study", ""], []
    fst = {c: load_features(c) for c in ["Radcure", "Maastro"]}
    flp = {}
    for w in ["lp", "lp2"]:
        cohort.FEATURE_DIRS["published"] = FEAT[w]
        flp[w] = {c: load_features(c) for c in ["Radcure", "Maastro"]}
    cohort.FEATURE_DIRS["published"] = FEAT["st"]

    L += ["## 1. Input features vs the paper's saved models (feature mean of the training split, in SDs)", "",
          "| Model | Endpoint | Cohort | |mean − saved| / SD (median) | SD ratio | verdict |", "|---|---|---|---|---|---|"]
    for ep in ["efs", "lr", "dr"]:                       # CT-FM-ST bundles (OS bundle has a broken scaler)
        for c in ["Radcure", "Maastro"]:
            try:
                mu_b, sd_b, _ = bundles.load_bundle(ep, c)
            except Exception as e:
                L.append(f"| CT-FM-ST | {ep} | {c} | — | — | no saved model ({str(e)[:40]}) |"); continue
            tr = split_dataset_ratio(build_cohort(c, ep))[0]
            X = np.stack([fst[c][k] for k in tr.key])
            dm, sr = float(np.median(np.abs(X.mean(0) - mu_b) / sd_b)), float(np.median(X.std(0) / sd_b))
            ok = dm < 0.01 and abs(sr - 1) < 0.01
            bad += [] if ok else [f"CT-FM-ST input {ep} {c}"]
            L.append(f"| CT-FM-ST | {ep} | {c} | {dm:.4f} | {sr:.4f} | {'match' if ok else 'MISMATCH'} |")
    for ep, d in lp_models.items():                      # CT-FM-LP bundles
        for c in ["Radcure", "Maastro"]:
            f = WS / d / c / "inference_fulltrain_segresnet_pretrained_frozen.pt"
            if not f.exists():
                L.append(f"| CT-FM-LP | {ep} | {c} | — | — | no saved model |"); continue
            sd = torch.load(f, map_location="cpu", weights_only=False); sd = sd.get("state_dict", sd)
            mu_b, sd_b = sd["scaler_mean"].numpy(), sd["scaler_scale"].numpy()
            tr = split_dataset_ratio(build_cohort(c, ep))[0]
            X = np.stack([flp[LP_INPUT[ep]][c][k] for k in tr.key])
            dm, sr = float(np.median(np.abs(X.mean(0) - mu_b) / sd_b)), float(np.median(X.std(0) / sd_b))
            ok = dm < 0.01 and abs(sr - 1) < 0.01
            bad += [] if ok else [f"CT-FM-LP input {ep} {c}"]
            L.append(f"| CT-FM-LP ({LP_INPUT[ep]}) | {ep} | {c} | {dm:.4f} | {sr:.4f} | {'match' if ok else 'MISMATCH'} |")

    L += ["", "## 2. CT-FM-ST C-index: this pipeline vs the paper's per-seed results (original data)", "",
          "| Endpoint | Cohort | Train size | Seed | Paper | This run | |diff| |", "|---|---|---|---|---|---|---|"]
    cells = [(ep, c, None, None) for ep in ["os", "efs", "lr", "dr"] for c in ["Radcure", "Maastro"]]
    cells += [("os", "Radcure", n, s) for n in [10, 50, 200] for s in range(5)]
    worst = 0.0
    for ep, c, n, s in cells:
        p = pd.read_csv(WS / paper_csv[ep]); p = p[p.Dataset == c]
        d = build_cohort(c, ep); tr, va, te = split_dataset_ratio(d)
        n_ = len(tr) if n is None else n
        sub = stratified_subsample(tr, n_, s if s is not None else 0)
        get = lambda df: np.stack([fst[c][k] for k in df.key])
        fit = fit_ctfm_st(sub, va, te, get(sub), get(va), get(te), eval_times_for(sub.time.values, sub.event.values))
        mine = cindex(te.time.values, te.event.values, fit.risk_test)
        q = p[(p.Train_Size == n_) & (p.Seed == (s if s is not None else 0))]["C-index"]
        paper = float(q.iloc[0]) if len(q) else float("nan")
        diff = abs(mine - paper)
        worst = max(worst, diff if np.isfinite(diff) else 0)
        L.append(f"| {ep} | {c} | {n_} | {s if s is not None else 'full'} | {paper:.4f} | {mine:.4f} | {diff:.4f} |")
        print(f"   {ep} {c} n={n_} seed={s}: paper {paper:.4f}  this run {mine:.4f}", flush=True)
    if worst > 0.002:
        bad.append(f"CT-FM-ST C-index differs from the paper by up to {worst:.4f}")
    L += ["", f"Largest CT-FM-ST difference: {worst:.4f}. " + ("**PASS**" if not bad else "**FAIL: " + "; ".join(bad) + "**")]
    (out / "verify.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("features"); p.add_argument("--which", choices=list(FEAT), required=True)
    p.add_argument("--force", action="store_true")
    p = sp.add_parser("verify"); p.add_argument("--out", required=True)
    p = sp.add_parser("segres-cache"); p.add_argument("--procs", type=int, default=16)
    p.add_argument("--which", choices=["lp", "lp2"], required=True)
    p.add_argument("--force", action="store_true")
    p = sp.add_parser("segres-worker"); p.add_argument("--data", choices=list(PAPER_CSV), required=True)
    p.add_argument("--endpoints", nargs="+", required=True); p.add_argument("--gpu", default="?")
    p.add_argument("--root", required=True)
    p = sp.add_parser("run"); p.add_argument("--data", choices=list(PAPER_CSV), required=True)
    p.add_argument("--features", choices=list(FEAT), default=None); p.add_argument("--root", required=True)
    p.add_argument("--seeds", type=int, default=5); p.add_argument("script", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    if a.cmd == "run" and a.script and a.script[0] == "--":
        a.script = a.script[1:]
    {"features": cmd_features, "verify": cmd_verify, "segres-cache": cmd_segres_cache,
     "segres-worker": cmd_segres_worker, "run": cmd_run}[a.cmd](a)
