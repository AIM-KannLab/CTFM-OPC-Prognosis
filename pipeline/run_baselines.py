"""Imaging-only baselines of the submitted paper, rerun on the revision protocol.

  spectre     SPECTRE: frozen ViT-Large CT foundation model (spectre-fm), features averaged
              over tokens, standardized on the training patients, then the SAME linear Cox
              head as CT-FM-LP -- so the two differ only in the frozen backbone. Features come
              from extract_spectre_features.py (HU-corrected input, GPU, once). The submitted
              run used 15 epochs, patience 8, batch 4 and 3 seeds to save time; here it gets
              the CT-FM-LP settings and 5 seeds. OS ONLY, as in the submitted paper; it is
              left out of subgroups and calibration, as in the paper too.

  ctfm_lp     CT-FM-LP: frozen CT-FM backbone, features standardized on the training
              patients, one linear Cox layer (no bias) trained with Adam -- lr 1e-4, batch 8,
              at most 40 epochs, early stopping on validation C-index with patience 15 --
              the configuration of few_shot_single_institution_pytorch_head_*.py. The
              backbone is frozen, so the layer trains on the cached HU CT-FM features, which
              are exactly what that backbone outputs. CPU only, minutes.
  segres_e2e  SegRes-E2E: trained by train_segres.py with A1_SEGRES_VARIANT=e2e (GPU). This
              script only reads the trained networks' outputs.

Corrections relative to the submitted runs, the same as for SegRes-FT:
  * the Cox loss counts only events (the original (B,1) x (B,) broadcast counted every
    censored patient as an event),
  * validation is scored in eval mode,
  * the input is the HU-corrected preprocessing, the same as CT-FM-ST's,
  * the data are the corrected files (pure LR / DM, fixed Radcure RFS times, N stage).

Protocol, identical to run_a1.py so the rows sit in the same Table S4: seed-42 70/15/15
split; the same shot grid and few-shot subsets; 5 seeds at every training size INCLUDING
full shot (both models are stochastic, as in the submitted table); early stopping on the
fixed validation split; scored on the held-out test set. Scores are centred on the
training patients' mean, and survival curves for the IBS use their Breslow baseline --
the same steps run_a1.py applies to SegRes-FT.

Full-shot models are kept: CT-FM-LP's head and scaler per seed in <model>/models/ (risk =
((features - scaler_mean) / scaler_scale) . weight); SegRes-E2E's networks stay in the
network store (model_info.json records where), one main.pt per full-shot seed.

Output: results_baselines_<tag>_<run>/<model>/predictions/, in the layout run_a1.py writes,
so make_table_s4.py, run_subgroup.py and make_figures_s4.py read it unchanged:
  <cohort>_<endpoint>_primary_fewshot_n<N>.npz   all seeds stacked: times, events, risk, surv
  <cohort>_<endpoint>_primary_fullshot.npz       the same, plus the test patients' keys
plus n_seeds and n_train in each file. Imaging-only models use no clinical variable, so
the "primary" files serve both clinical specs. <model>/training_log.csv records every fit.
A cell whose file already exists is skipped, so a stopped run resumes (--force redoes it).

    python run_baselines.py --model ctfm_lp --run <id>
    A1_SEGRES_VARIANT=e2e python run_baselines.py --model segres_e2e --run <id>
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import segres_cache as SC
from cohort import (COHORTS, DATA_VERSION, ENDPOINT_PAPER_NAME, ENDPOINTS, FEATURE_VERSION,
                    NSTAGE, SEGRES_DIR, SEGRES_PRECISION, SEGRES_VARIANT, build_cohort,
                    new_run_id, results_dir, split_dataset_ratio, tee_output, write_run_info)
from ipcw_brier import eval_times_for
from models import _cindex
from subsample import SEEDS, shot_grid, stratified_subsample

SPEC_TAG = "primary"
LP = {"batch_size": 8, "lr": 1e-4, "max_epochs": 40, "patience": 15, "min_delta": 1e-4}
LABEL = {"ctfm_lp": "CT-FM-LP", "segres_e2e": "SegRes-E2E", "spectre": "SPECTRE"}
FROZEN_FEATURES = {"ctfm_lp", "spectre"}          # linear Cox head on frozen features
MODEL_ENDPOINTS = {"spectre": ["os"]}              # SPECTRE: OS only, as in the submitted paper


def train_linear_cox(f_tr, t_tr, e_tr, f_va, t_va, e_va, seed):
    """Linear Cox head on frozen features (CT-FM-LP, SPECTRE).
    Returns (predict(features) -> log-hazard, record, parameters)."""
    import torch
    import torch.nn as nn
    from sklearn.preprocessing import StandardScaler
    from train_segres import cox_loss            # the corrected loss, shared with SegRes

    torch.manual_seed(seed)
    np.random.seed(seed)
    scaler = StandardScaler().fit(f_tr)          # FrozenLinearCoxNet's scaler buffers
    as_t = lambda a: torch.as_tensor(np.asarray(a, float), dtype=torch.float32)
    X, Xv = as_t(scaler.transform(f_tr)), as_t(scaler.transform(f_va))
    t, e = as_t(t_tr), as_t(e_tr)
    head = nn.Linear(X.shape[1], 1, bias=False)
    opt = torch.optim.Adam(head.parameters(), lr=LP["lr"])
    gen = torch.Generator().manual_seed(seed)
    best_c, wait, best_w, best_epoch, epochs_run = 0.0, 0, None, None, 0
    for epoch in range(LP["max_epochs"]):
        perm = torch.randperm(len(X), generator=gen)
        for b in range(0, len(X), LP["batch_size"]):
            idx = perm[b:b + LP["batch_size"]]
            opt.zero_grad()
            loss = cox_loss(head(X[idx]), t[idx], e[idx])
            loss.backward()
            opt.step()
        epochs_run = epoch + 1
        with torch.no_grad():
            c = _cindex(t_va, e_va, head(Xv).reshape(-1).numpy())
        if c > best_c + LP["min_delta"]:
            best_c, wait, best_epoch = float(c), 0, epoch + 1
            best_w = head.weight.detach().clone()
        else:
            wait += 1
            if wait >= LP["patience"]:
                break
    if best_w is not None:
        with torch.no_grad():
            head.weight.copy_(best_w)

    def predict(F):
        with torch.no_grad():
            return head(as_t(scaler.transform(F))).reshape(-1).numpy().astype(float)

    params = {"weight": head.weight.detach().numpy().reshape(-1).copy(),
              "scaler_mean": scaler.mean_, "scaler_scale": scaler.scale_}
    return predict, {"best_epoch": best_epoch or epochs_run, "epochs_run": epochs_run,
                     "best_val_c": best_c if best_w is not None else float("nan")}, params


def segres_e2e_scores(cohort, endpoint, train, val, test, seed):
    """Raw outputs of the SegRes-E2E network trained on exactly these patients."""
    e = SC.load_entry(cohort, endpoint, train, val, seed)
    return (e.raw(cohort, train["key"].values), e.raw(cohort, test["key"].values),
            {"best_epoch": e.best_epoch, "epochs_run": e.epochs_run, "best_val_c": e.best_val_c})


def run_cell(model, co, ep, out: Path, force: bool, log: list):
    df = build_cohort(co, ep)
    tr_full, va, te = split_dataset_ratio(df)
    eval_times = eval_times_for(tr_full["time"].values, tr_full["event"].values)
    t_te, e_te = te["time"].values, te["event"].values
    if model in FROZEN_FEATURES:
        if model == "ctfm_lp":
            from extract_features import load_features
            fmap = load_features(co)
        else:
            from extract_spectre_features import load_spectre_features
            fmap = load_spectre_features(co, FEATURE_VERSION)
        get = lambda d: np.stack([fmap[k] for k in d["key"]])
        f_va = get(va)
    else:
        SC.check_trained([co], [ep], ("full", "few"))
    sizes = [s for s in shot_grid(len(tr_full)) if s < len(tr_full)] + ["full"]
    print(f"\n[{co}/{ENDPOINT_PAPER_NAME[ep]}] train={len(tr_full)} val={len(va)} "
          f"test={len(te)} ({int(e_te.sum())} events)   sizes {sizes}")
    for size in sizes:
        name = (f"{co}_{ep}_{SPEC_TAG}_fullshot.npz" if size == "full"
                else f"{co}_{ep}_{SPEC_TAG}_fewshot_n{size}.npz")
        path = out / name
        if path.exists() and not force:
            print(f"   n={size}: done -> skipped")
            continue
        risks, survs, cs, t0 = [], [], [], time.time()
        for seed in SEEDS:
            train = tr_full if size == "full" else stratified_subsample(tr_full, size, seed)
            if model in FROZEN_FEATURES:
                predict, rec, params = train_linear_cox(get(train), train["time"].values,
                                                     train["event"].values, f_va, va["time"].values,
                                                     va["event"].values, seed)
                r_tr, r_te = predict(get(train)), predict(get(te))
                if size == "full":       # a reusable model: risk = ((x - mean) / scale) . weight
                    mdir = out.parent / "models"; mdir.mkdir(exist_ok=True)
                    np.savez(mdir / f"{co}_{ep}_fullshot_seed{seed}.npz", **params,
                             features=FEATURE_VERSION, data=DATA_VERSION,
                             best_epoch=rec["best_epoch"])
            else:
                r_tr, r_te, rec = segres_e2e_scores(co, ep, train, va, te, seed)
            ok = np.isfinite(r_tr).all() and np.isfinite(r_te).all()
            c = _cindex(t_te, e_te, r_te) if ok else float("nan")
            log.append({"cohort": co, "endpoint": ENDPOINT_PAPER_NAME[ep], "n_train": len(train),
                        "size": size, "seed": seed, "test_cindex": c, "failed": not ok, **rec})
            if not ok:
                continue
            mu = float(r_tr.mean())
            risks.append(r_te - mu)
            survs.append(SC.breslow_survival(r_tr - mu, train["time"].values,
                                             train["event"].values, r_te - mu, eval_times))
            cs.append(c)
        k = len(risks)
        if k == 0:
            print(f"   n={size}: every seed failed -- no file written")
            continue
        arrays = dict(times=np.tile(t_te, k), events=np.tile(e_te, k),
                      risk=np.concatenate(risks), surv=np.vstack(survs), eval_times=eval_times,
                      n_seeds=k, n_train=len(tr_full) if size == "full" else int(size))
        if size == "full":
            arrays["keys"] = te["key"].values          # one copy; risk holds k seeds stacked
        tmp = path.with_name(path.stem + ".tmp.npz")
        np.savez_compressed(tmp, **arrays)
        tmp.replace(path)
        print(f"   n={size}: C = {np.mean(cs):.3f} ± {np.std(cs, ddof=1) if k > 1 else 0:.3f} "
              f"over {k} seeds  ({time.time() - t0:.0f}s)", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", choices=list(LABEL), required=True)
    ap.add_argument("--run", default=None, help="run id YYYYMMDD_HHMMSS; default = new run")
    ap.add_argument("--cohort", choices=list(COHORTS), default=None)
    ap.add_argument("--endpoint", choices=ENDPOINTS, default=None)
    ap.add_argument("--force", action="store_true", help="redo cells already on disk")
    ap.add_argument("--out", default=None, help="run folder (default: results_baselines_<tag>_<run>)")
    a = ap.parse_args()
    if a.model == "segres_e2e" and SEGRES_VARIANT != "e2e":
        raise SystemExit("segres_e2e reads the SegRes-E2E networks: set A1_SEGRES_VARIANT=e2e")
    import torch
    import train_segres as TS                    # the loss the linear head trains with (TS.cox_loss)
    torch.set_num_threads(4)
    run_id = a.run or new_run_id()
    root = Path(a.out) if a.out else results_dir("results_baselines", run_id)
    out = root / a.model / "predictions"
    out.mkdir(parents=True, exist_ok=True)
    tee_output(root / "logs" / f"{a.model}_{a.cohort or 'all'}_{a.endpoint or 'all'}.log")
    write_run_info(root, script="run_baselines.py")
    info = {"model": LABEL[a.model], "data": DATA_VERSION, "input": FEATURE_VERSION,
            "nstage": NSTAGE, "seeds": SEEDS, "endpoints": MODEL_ENDPOINTS.get(a.model, ENDPOINTS),
            "command": " ".join(sys.argv)}
    if a.model in FROZEN_FEATURES:
        backbone = ("frozen CT-FM (512 features)" if a.model == "ctfm_lp" else
                    "frozen SPECTRE ViT-Large, features averaged over tokens")
        info["config"] = {**LP, "head": f"linear, no bias, on standardized features of the {backbone}",
                          "loss": TS.loss_label()}
    else:
        info["networks"] = str(SEGRES_DIR)
        info["config"] = {"variant": SEGRES_VARIANT, "precision": SEGRES_PRECISION}
    (root / a.model / "model_info.json").write_text(json.dumps(info, indent=1))
    print(f"command: {' '.join(sys.argv)}\nmodel: {LABEL[a.model]}   data: {DATA_VERSION}   "
          f"input: {FEATURE_VERSION}   N stage: {NSTAGE}\nrun {run_id} -> {root}")

    log = []
    endpoints = MODEL_ENDPOINTS.get(a.model, ENDPOINTS)
    if a.endpoint and a.endpoint not in endpoints:
        raise SystemExit(f"{LABEL[a.model]} is run for {endpoints} only")
    for co in ([a.cohort] if a.cohort else list(COHORTS)):
        for ep in ([a.endpoint] if a.endpoint else endpoints):
            run_cell(a.model, co, ep, out, a.force, log)
    if log:
        lp = root / a.model / "training_log.csv"
        old = pd.read_csv(lp) if lp.exists() else pd.DataFrame()
        pd.concat([old, pd.DataFrame(log)], ignore_index=True).to_csv(lp, index=False)
    print(f"\ndone -> {out}")


if __name__ == "__main__":
    main()
