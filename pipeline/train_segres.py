"""SegRes-FT for the revision analysis: train the networks once, keep their outputs.

SegRes-FT is the CT-FM architecture (SegResEncoder) trained FROM SCRATCH -- random
init, every weight trained -- with an MLP Cox head: the "SegRes-FT" entry of
few_shot_single_institution_{,lr_,dr_,efs_}running.py, with that code's configuration
(identical in all four scripts):

    backbone   SegResEncoder(blocks_down=(1, 2, 2, 4, 4)), random init  (77.8 M weights)
    head       512 -> 256 -> LayerNorm -> ReLU -> Dropout(0.3) -> 128 -> 1
    loss       negative Cox partial log-likelihood, averaged over the events in a batch
    optimiser  Adam, lr 1e-5 backbone / 1e-4 head, batch 8, at most 40 epochs
    stopping   validation C-index after every epoch, patience 15, min delta 1e-4;
               the best epoch's weights are kept

Two defects of that code are corrected, because both change what the network learns:
  * the loss: the head outputs shape (B, 1) and the event indicator has shape (B,), so
    `uncensored * events` broadcasts to (B, B) and every censored patient in a batch is
    counted as an event. Here the output is flattened first.
  * validation was scored in training mode (dropout on, BatchNorm using and updating
    batch statistics); here in eval mode, as the test set always was.

Everything else follows the revision protocol, so the networks drop into the same
analysis as CT-FM-ST (run_a1.py, run_external.py with A1_IMAGING=segres_ft):
  * the same patients: seed-42 70/15/15 split, shot grid and few-shot subsets, 5 seeds;
  * the same input as the revision's CT-FM features (A1_FEATURES; use `hu`), padded
    back to 64 x 160 x 160 with air so that scans can be batched;
  * the combined model's imaging score for the TRAINING patients comes from
    out-of-fold networks, on the same folds as CT-FM-ST's stage 1 (5-fold; leave-one-
    out below 25 patients). Each is trained for the number of epochs the main network
    chose on validation, as CT-FM-ST's folds reuse its validation-chosen ridge penalty;
  * the network trained on a cohort's full training split also scores every patient of
    the other cohort: that is the external validation (Radcure -> Maastro).

SegRes-E2E (A1_SEGRES_VARIANT=e2e) is the submitted paper's from-scratch baseline, from
few_shot_single_institution_pytorch_head_*.py: the same backbone and training settings, but
a single linear layer on the pooled features (LinearCoxNet) and BatchNorm kept frozen while
training. It gets the same two corrections as SegRes-FT and the same input. It is an
imaging-only baseline: no out-of-fold networks, and 5 seeds at every training size
including full shot (segres_cache.FULL_SHOT_SEEDS). run_baselines.py reads it.

One task per network. Any number of workers -- one per GPU -- share the task list
through claim files, so no network is trained twice, and a stopped run resumes where it
stopped (a network that was interrupted is retrained from its start). GPU training is
not bit-reproducible from run to run; the trained networks are kept, so every analysis
of them is.

    python train_segres.py --cache-volumes          # once: preprocess every scan (CPU)
    CUDA_VISIBLE_DEVICES=0 python train_segres.py --regime full   # a worker; one per GPU
    python train_segres.py --status                 # what is done, running, left

run_pipeline_segres.sh does all of this, then the analysis, in one tmux session.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import random
import socket
import sys
import time
import traceback
from contextlib import contextmanager
from datetime import datetime
from multiprocessing import Pool

import numpy as np
import pandas as pd

# less GPU memory lost to fragmentation (must be set before torch touches the GPU)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch
import torch.nn as nn
import torch.nn.functional as F

import segres_cache as SC
from cohort import (COHORTS, DATA_VERSION, ENDPOINTS, FEATURE_VERSION, SEGRES_DIR,
                    SEGRES_EPOCHS, SEGRES_PRECISION, SEGRES_VARIANT, VOLUME_DIR, build_cohort)
from models import _cindex

SPATIAL = (64, 160, 160)          # stored scan size (D, H, W after Orientation "SPL")
ARCH = {"blocks_down": (1, 2, 2, 4, 4)}
TRAIN = {"batch_size": 8, "lr": 1e-4, "backbone_lr": 1e-5, "max_epochs": SEGRES_EPOCHS,
         "patience": 15, "min_delta": 1e-4}
EVAL_BATCH = 8                    # as training: keeps GPU memory predictable
AMP = SEGRES_PRECISION == "bf16"
POLL = 30                         # seconds between looks at the task list when blocked


def _now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _atomic_json(path, obj):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, path)


def _atomic_npz(path, **arrays):
    tmp = path.with_name(path.stem + ".tmp.npz")      # np.savez insists on .npz
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


# =========================================================================== scans

def _vol_paths(cohort):
    return VOLUME_DIR / f"{cohort.lower()}.npy", VOLUME_DIR / f"{cohort.lower()}.json"


_BASE = _PAD = None


def _init_prep():
    global _BASE, _PAD
    torch.set_num_threads(1)
    from monai.transforms import ResizeWithPadOrCrop
    from extract_features import build_transform
    _BASE = build_transform(FEATURE_VERSION)        # exactly the CT-FM feature input
    _PAD = ResizeWithPadOrCrop(SPATIAL, mode="constant")   # pads with 0 = air


def _prep(path):
    x = _BASE(str(path))
    shape = tuple(int(s) for s in x.shape[1:])
    return np.asarray(_PAD(x), dtype=np.float32), shape


def cache_volumes(procs: int, force: bool) -> None:
    """Preprocess every scan once into one float32 array per cohort (~6.5 MB a scan).

    Applying the transform takes seconds per scan; training reads every scan up to
    40 times per network, so it reads this array instead.
    """
    VOLUME_DIR.mkdir(parents=True, exist_ok=True)
    for c in COHORTS:
        npy, meta = _vol_paths(c)
        if npy.exists() and meta.exists() and not force:
            print(f"{c}: {npy} exists -> skipped (--force to rebuild)")
            continue
        df = build_cohort(c, "os")                  # every patient; endpoint irrelevant
        tmp = npy.with_name(npy.stem + ".tmp.npy")
        arr = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float32,
                                        shape=(len(df), 1, *SPATIAL))
        print(f"{c}: preprocessing {len(df)} scans ({FEATURE_VERSION}) with {procs} processes "
              f"-> {npy}")
        shapes, t0 = [], time.time()
        with Pool(procs, initializer=_init_prep) as pool:
            for i, (x, shp) in enumerate(pool.imap(_prep, df["path"].tolist(), chunksize=2)):
                arr[i] = x
                shapes.append(shp)
                if (i + 1) % 100 == 0 or i + 1 == len(df):
                    print(f"   {i + 1}/{len(df)}  ({time.time() - t0:.0f}s)", flush=True)
        if not np.isfinite(arr).all():
            raise RuntimeError(f"{c}: non-finite values in the preprocessed scans")
        lo, hi = float(arr.min()), float(arr.max())
        arr.flush()
        del arr
        os.replace(tmp, npy)
        cropped = sum(any(s > S for s, S in zip(shp, SPATIAL)) for shp in shapes)
        _atomic_json(meta, {
            "cohort": c, "preproc": FEATURE_VERSION, "keys": df["key"].tolist(),
            "shape": [len(df), 1, *SPATIAL], "dtype": "float32", "value_range": [lo, hi],
            "transform": f"extract_features.build_transform('{FEATURE_VERSION}') + "
                         f"ResizeWithPadOrCrop({SPATIAL}, constant 0)",
            "shapes_before_padding": shapes, "scans_larger_than_target": cropped,
            "created": _now()})
        print(f"   saved: values {lo:.4f}..{hi:.4f}; "
              f"{sum(s != SPATIAL for s in shapes)} scans padded to {SPATIAL}, "
              f"{cropped} had to be cropped")


class Volumes:
    """One cohort's preprocessed scans, addressed by patient key, held in GPU memory.

    Reading batches from disk starves the GPU on this busy shared machine (~100 MB/s,
    under 1 batch/s), so on first use each worker copies the cohort's whole array to
    its GPU once -- Radcure 8.8 GB, Maastro 1.1 GB, float32 as stored -- and builds
    every batch there.
    """

    def __init__(self, cohort, device):
        npy, meta = _vol_paths(cohort)
        if not (npy.exists() and meta.exists()):
            raise SystemExit(f"no preprocessed scans for {cohort} in {VOLUME_DIR} -- "
                             f"run first:  python train_segres.py --cache-volumes")
        m = json.loads(meta.read_text())
        if m["preproc"] != FEATURE_VERSION:
            raise SystemExit(f"{npy} holds '{m['preproc']}' scans, but A1_FEATURES is "
                             f"'{FEATURE_VERSION}'")
        self.cohort, self.device = cohort, device
        self._disk = np.load(npy, mmap_mode="r")
        self._gpu = None
        self.keys = list(m["keys"])
        self.index = {k: i for i, k in enumerate(self.keys)}

    def rows(self, keys):
        miss = [k for k in keys if k not in self.index]
        if miss:
            raise SystemExit(f"{len(miss)} patients have no preprocessed scan (e.g. {miss[0]}) "
                             f"-- rebuild:  python train_segres.py --cache-volumes --force")
        return np.asarray([self.index[k] for k in keys])

    def get(self, rows) -> torch.Tensor:
        if self._gpu is None:                    # first use: copy the whole cohort, once
            t0 = time.time()
            g = torch.empty(self._disk.shape, dtype=torch.float32, device=self.device)
            for b in range(0, len(g), 32):
                g[b:b + 32] = torch.from_numpy(np.array(self._disk[b:b + 32]))
            self._gpu = g
            print(f"   {self.cohort}: {len(g)} scans copied to GPU memory "
                  f"({g.numel() * 4 / 1e9:.1f} GB, {time.time() - t0:.0f}s)", flush=True)
        return self._gpu[torch.as_tensor(np.asarray(rows), device=self.device)]


# =========================================================================== model

class SurvivalNet(nn.Module):
    """Verbatim from few_shot_single_institution_dr_running.py:151-179."""

    def __init__(self, backbone: nn.Module, feature_dim: int = 512, hidden_dim1: int = 256,
                 hidden_dim2: int = 128, dropout_p: float = 0.3):
        super().__init__()
        self.backbone = backbone
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim1),
            nn.LayerNorm(hidden_dim1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim1, hidden_dim2),
        )
        self.out = nn.Linear(hidden_dim2, 1)

    def forward(self, x):
        feats = F.adaptive_avg_pool3d(self.backbone(x)[-1], 1).flatten(1)
        return self.out(self.mlp(feats))


class LinearCoxNet(nn.Module):
    """SegRes-E2E head, verbatim from few_shot_single_institution_pytorch_head_*.py:
    backbone, global average pool, one linear layer without bias."""

    def __init__(self, backbone: nn.Module, feature_dim: int = 512):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(feature_dim, 1, bias=False)

    def forward(self, x):
        return self.head(F.adaptive_avg_pool3d(self.backbone(x)[-1], 1).flatten(1))


def build_model() -> nn.Module:
    from lighter_zoo import SegResEncoder
    backbone = SegResEncoder(**ARCH)                # random init, every weight trained
    return LinearCoxNet(backbone) if SEGRES_VARIANT == "e2e" else SurvivalNet(backbone)


def _freeze_batchnorm(model: nn.Module) -> None:
    """SegRes-E2E keeps BatchNorm in eval mode during training, as the original script did
    (its train_model sets every _BatchNorm to eval at the start of each epoch). With random
    init the running statistics stay at mean 0 / var 1; the affine weights still train."""
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.eval()


def cox_loss(preds, times, events, eps: float = 1e-7):
    """The original CoxPHLoss, with the output flattened first.

    Without the flattening, (B, 1) * (B,) broadcasts to (B, B): the loss then sums the
    partial-likelihood term of EVERY patient, censored or not.
    """
    preds = preds.reshape(-1)
    order = torch.argsort(times, descending=True)
    preds = preds[order] - preds.max()
    events = events[order]
    log_risk = torch.log(torch.cumsum(torch.exp(preds), dim=0) + eps)
    n_events = events.sum()
    return -(((preds - log_risk) * events).sum() / (n_events if n_events > 0 else 1.0))


cox_loss.label = "Cox partial likelihood, events only (train_segres.cox_loss)"


def loss_label() -> str:
    """The loss that actually runs: runner.py may replace cox_loss, so run records name it from here."""
    return getattr(cox_loss, "label", getattr(cox_loss, "__name__", str(cox_loss)))


@torch.no_grad()
def predict(model, vol: Volumes, rows, device) -> np.ndarray:
    """Network output (log-hazard) in eval mode; the model's mode is restored after."""
    was_training = model.training
    model.eval()
    out = []
    for b in range(0, len(rows), EVAL_BATCH):
        x = vol.get(rows[b:b + EVAL_BATCH])
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=AMP):
            out.append(model(x).float().reshape(-1).cpu().numpy())
    model.train(was_training)
    return np.concatenate(out) if out else np.empty(0)


def train_network(vol, rows, times, events, seed, device, epochs, val=None, tag=""):
    """Train one SegRes-FT network.

    val = (rows, times, events): early stopping on the validation C-index (main network).
    val = None: exactly `epochs` epochs, no validation (out-of-fold networks).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    model = build_model().to(device)
    optim = torch.optim.Adam([
        {"params": [p for p in model.backbone.parameters() if p.requires_grad],
         "lr": TRAIN["backbone_lr"]},
        {"params": [p for n, p in model.named_parameters() if not n.startswith("backbone")],
         "lr": TRAIN["lr"]},
    ])
    gen = torch.Generator().manual_seed(seed)
    t_all = torch.as_tensor(np.asarray(times, float), dtype=torch.float32)
    e_all = torch.as_tensor(np.asarray(events, float), dtype=torch.float32)
    n, B = len(rows), TRAIN["batch_size"]
    best_c, wait, best_state, best_epoch = 0.0, 0, None, None
    curve, losses, epochs_run, t0 = [], [], 0, time.time()
    for epoch in range(epochs):
        model.train()
        if SEGRES_VARIANT == "e2e":
            _freeze_batchnorm(model)
        total = 0.0
        perm = torch.randperm(n, generator=gen)
        for b in range(0, n, B):
            idx = perm[b:b + B]
            x = vol.get(rows[idx.numpy()])
            optim.zero_grad()
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=AMP):
                out = model(x)
            loss = cox_loss(out.float(), t_all[idx].to(device), e_all[idx].to(device))
            loss.backward()
            optim.step()
            total += float(loss) * len(idx)
        losses.append(total / n)
        epochs_run = epoch + 1
        if val is None:
            continue
        c = _cindex(val[1], val[2], predict(model, vol, val[0], device))
        curve.append(float(c))
        improved = c > best_c + TRAIN["min_delta"]
        if improved:
            best_c, wait, best_epoch = float(c), 0, epoch + 1
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
        print(f"   {tag} epoch {epoch + 1:02d}  loss {losses[-1]:.4f}  val C {c:.4f}"
              f"{'  *' if improved else ''}  ({time.time() - t0:.0f}s)", flush=True)
        if not improved and wait >= TRAIN["patience"]:
            print(f"   {tag} early stop: best epoch {best_epoch}, val C {best_c:.4f}")
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, {"best_epoch": best_epoch or epochs_run, "epochs_run": epochs_run,
                   "best_val_c": best_c if best_state is not None else float("nan"),
                   "val_curve": curve, "train_loss": losses,
                   "seconds": round(time.time() - t0, 1)}


# =========================================================================== tasks

def _tag(ts, k=None):
    name = f"{ts.cohort}/{ts.endpoint} n{len(ts.train)} s{ts.seed}"
    return f"[{name} main]" if k is None else f"[{name} fold {k + 1}/{ts.K}]"


def run_main(ts, vols, device, gpu):
    vol = vols[ts.cohort]
    tr, va, te = ts.train, ts.val, ts.test
    print(f"\n{_now()}  {_tag(ts)}  train {len(tr)} ({int(tr.event.sum())} events), "
          f"early stopping on val {len(va)} ({int(va.event.sum())} events)  GPU {gpu}", flush=True)
    model, rec = train_network(vol, vol.rows(tr.key.tolist()), tr.time.values, tr.event.values,
                               ts.seed, device, TRAIN["max_epochs"],
                               val=(vol.rows(va.key.tolist()), va.time.values, va.event.values),
                               tag=_tag(ts))
    scored = {}
    own = pd.concat([tr, va, te]).key.tolist()
    scored[ts.cohort] = (own, predict(model, vol, vol.rows(own), device))
    if ts.other:                         # external validation: every patient of the other cohort
        ov = vols[ts.other]
        scored[ts.other] = (ov.keys, predict(model, ov, np.arange(len(ov.keys)), device))
    arrays = {}
    for c, (k, r) in scored.items():
        arrays[f"keys_{c}"], arrays[f"risk_{c}"] = np.asarray(k), r
    _atomic_npz(ts.dir / "main.npz", **arrays)
    if ts.regime == "full":                  # every full-shot network is kept (311 MB each)
        tmp = ts.dir / "main.tmp.pt"
        torch.save(model.state_dict(), tmp)
        os.replace(tmp, ts.dir / "main.pt")
    r_tr = scored[ts.cohort][1][:len(tr)]
    _atomic_json(ts.dir / "main.json", {
        "model": "SegRes-E2E" if SEGRES_VARIANT == "e2e" else "SegRes-FT",
        "cohort": ts.cohort, "endpoint": ts.endpoint,
        "regime": ts.regime, "seed": ts.seed, "n_train": len(tr),
        "events_train": int(tr.event.sum()), "n_val": len(va),
        "fingerprint": SC.fingerprint(tr, va), "n_folds": ts.K, **rec,
        "test_c": float(_cindex(te.time.values, te.event.values,
                                scored[ts.cohort][1][len(tr) + len(va):])),
        "train_output_mean": float(r_tr.mean()), "train_output_sd": float(r_tr.std()),
        "scored": {c: len(k) for c, (k, _) in scored.items()},
        "n_params": int(sum(p.numel() for p in model.parameters())),
        "config": {"arch": {"blocks_down": list(ARCH["blocks_down"])}, **TRAIN,
                   "head": ("linear 512-1, no bias; BatchNorm frozen" if SEGRES_VARIANT == "e2e"
                            else "512-256-LN-ReLU-Drop0.3-128-1"),
                   "variant": SEGRES_VARIANT, "precision": SEGRES_PRECISION,
                   "input": FEATURE_VERSION, "data": DATA_VERSION,
                   "loss": loss_label(), "validation": "eval mode"},
        "gpu": torch.cuda.get_device_name(device), "torch": torch.__version__,
        "finished": _now()})
    print(f"{_now()}  {_tag(ts)} done: best epoch {rec['best_epoch']}/{rec['epochs_run']}, "
          f"val C {rec['best_val_c']:.4f}  ({rec['seconds'] / 60:.1f} min)", flush=True)


def run_fold(ts, k, vols, device, gpu):
    meta = json.loads((ts.dir / "main.json").read_text())
    epochs = int(meta["best_epoch"])
    fit_idx, out_idx = ts.folds[k]
    vol = vols[ts.cohort]
    fit, held = ts.train.iloc[fit_idx], ts.train.iloc[out_idx]
    base = dict(keys=held.key.values.astype(str), epochs=epochs, n_fit=len(fit),
                events_fit=int(fit.event.sum()), fingerprint=meta["fingerprint"])
    path = ts.dir / f"fold{k}of{ts.K}.npz"
    if fit.event.sum() < 2:              # as CT-FM-ST's stage 1: too few events to fit
        _atomic_npz(path, risk=np.full(len(held), np.nan), fit_mean=np.nan, fit_sd=np.nan,
                    failed=True, seconds=0.0, **base)
        print(f"{_now()}  {_tag(ts, k)} skipped: {int(fit.event.sum())} events in its "
              f"training part -> its patients get average risk", flush=True)
        return
    t0 = time.time()
    model, rec = train_network(vol, vol.rows(fit.key.tolist()), fit.time.values,
                               fit.event.values, ts.seed, device, epochs, tag=_tag(ts, k))
    r_fit = predict(model, vol, vol.rows(fit.key.tolist()), device)
    r_out = predict(model, vol, vol.rows(held.key.tolist()), device)
    _atomic_npz(path, risk=r_out, fit_mean=float(r_fit.mean()), fit_sd=float(r_fit.std()),
                failed=False, seconds=round(time.time() - t0, 1), **base)
    print(f"{_now()}  {_tag(ts, k)} done: {epochs} epochs on {len(fit)} patients, "
          f"{len(held)} held out  ({(time.time() - t0) / 60:.1f} min)  GPU {gpu}", flush=True)


def _task_name(k):
    return "main" if k is None else f"fold{k}"


def _done(ts, k) -> bool:
    if k is None:
        return (ts.dir / "main.json").exists()
    return (ts.dir / f"fold{k}of{ts.K}.npz").exists()


@contextmanager
def _queue_lock():
    SEGRES_DIR.mkdir(parents=True, exist_ok=True)
    with open(SEGRES_DIR / ".queue.lock", "w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def _claim_alive(path) -> bool:
    """A claim is live while its worker process runs (same machine) -- else stale."""
    try:
        info = json.loads(path.read_text())
    except Exception:
        return False
    if info.get("host") != socket.gethostname():
        return True                       # another machine: cannot check, assume live
    try:
        os.kill(int(info["pid"]), 0)
        return True
    except (ProcessLookupError, ValueError, KeyError):
        return False
    except PermissionError:
        return True


def _try_claim(ts, k, gpu) -> bool:
    name = _task_name(k)
    if _done(ts, k) or (ts.dir / f".failed_{name}").exists():
        return False
    if k is not None and not _done(ts, None):
        return False                      # needs the main network's epoch count
    with _queue_lock():
        if _done(ts, k):
            return False
        claim = ts.dir / f".claim_{name}"
        if claim.exists():
            if _claim_alive(claim):
                return False
            print(f"{_now()}  {_tag(ts, k)} was interrupted -> retrained from its start")
        ts.dir.mkdir(parents=True, exist_ok=True)
        claim.write_text(json.dumps({"host": socket.gethostname(), "pid": os.getpid(),
                                     "gpu": gpu, "started": _now()}))
        return True


def _task_list(regimes, cohorts, endpoints):
    sets = list(SC.training_sets(cohorts, endpoints, regimes))
    for ts in sets:
        ts.K = len(ts.folds)
    # every main network first: its out-of-fold networks need its epoch count
    return sets, [(ts, None) for ts in sets] + [(ts, k) for ts in sets for k in range(ts.K)]


def _count(tasks):
    done = sum(_done(ts, k) for ts, k in tasks)
    failed = [(ts, k) for ts, k in tasks
              if not _done(ts, k) and (ts.dir / f".failed_{_task_name(k)}").exists()]
    running = sum(1 for ts, k in tasks if not _done(ts, k)
                  and (ts.dir / f".claim_{_task_name(k)}").exists()
                  and _claim_alive(ts.dir / f".claim_{_task_name(k)}"))
    return done, running, failed


def worker(regimes, cohorts, endpoints, gpu) -> int:
    if not torch.cuda.is_available():
        raise SystemExit("no GPU visible -- set CUDA_VISIBLE_DEVICES to a free GPU")
    device = torch.device("cuda")
    torch.set_num_threads(4)
    vols = {c: Volumes(c, device) for c in COHORTS}   # copied to the GPU on first use
    sets, tasks = _task_list(regimes, cohorts, endpoints)
    print(f"{_now()}  worker pid {os.getpid()} on GPU {gpu} ({torch.cuda.get_device_name(device)})"
          f"\n   regime {'+'.join(regimes)}: {len(sets)} training sets, {len(tasks)} networks"
          f"\n   model SegRes-{SEGRES_VARIANT.upper()}, input {FEATURE_VERSION}, data {DATA_VERSION}, "
          f"precision {SEGRES_PRECISION}, "
          f"max {TRAIN['max_epochs']} epochs  ->  {SEGRES_DIR}", flush=True)
    last = None
    while True:
        job = next(((ts, k) for ts, k in tasks if _try_claim(ts, k, gpu)), None)
        if job is None:
            done, running, failed = _count(tasks)
            if done + len(failed) == len(tasks):
                break
            state = f"{done}/{len(tasks)} done, {running} training on other workers"
            if state != last:
                print(f"{_now()}  nothing free to start ({state}) -- waiting", flush=True)
                last = state
            time.sleep(POLL)
            continue
        ts, k = job
        name = _task_name(k)
        try:
            (run_main(ts, vols, device, gpu) if k is None else run_fold(ts, k, vols, device, gpu))
        except BaseException as exc:
            if not isinstance(exc, KeyboardInterrupt):
                (ts.dir / f".failed_{name}").write_text(f"{_now()} GPU {gpu}\n{traceback.format_exc()}")
                print(f"!! {_tag(ts, k)} FAILED -- traceback in {ts.dir}/.failed_{name}")
            raise
        finally:
            (ts.dir / f".claim_{name}").unlink(missing_ok=True)
            torch.cuda.empty_cache()
    done, _, failed = _count(tasks)
    print(f"\n{_now()}  all networks for regime {'+'.join(regimes)} finished: {done}/{len(tasks)} "
          f"trained" + (f", {len(failed)} FAILED (e.g. {_tag(*failed[0])})" if failed else ""))
    return 1 if failed else 0


def status(regimes, cohorts, endpoints, brief=False) -> None:
    sets, tasks = _task_list(regimes, cohorts, endpoints)
    done, running, failed = _count(tasks)
    secs = 0.0
    for ts, k in tasks:
        if not _done(ts, k):
            continue
        if k is None:
            secs += json.loads((ts.dir / "main.json").read_text()).get("seconds", 0.0)
        else:
            secs += float(np.load(ts.dir / f"fold{k}of{ts.K}.npz")["seconds"])
    line = (f"{'SegRes-E2E' if SEGRES_VARIANT == 'e2e' else 'SegRes-FT'} {'+'.join(regimes)}: {done}/{len(tasks)} networks trained, {running} "
            f"training now, {len(failed)} failed; GPU time so far {secs / 3600:.1f} h")
    if brief:
        print(line)
        return
    print(f"{SEGRES_DIR}\n{line}\n")
    print(f"{'cohort':<8} {'endpoint':<8} {'regime':<6} {'sets':>5} {'main done':>10} {'folds done':>11}")
    keyf = lambda ts: (ts.cohort, ts.endpoint, ts.regime)
    for key in sorted({keyf(ts) for ts in sets}, key=lambda x: (x[0], ENDPOINTS.index(x[1]), x[2])):
        g = [ts for ts in sets if keyf(ts) == key]
        m = sum(_done(ts, None) for ts in g)
        f_tot = sum(ts.K for ts in g)
        f_done = sum(_done(ts, k) for ts in g for k in range(ts.K))
        print(f"{key[0]:<8} {key[1]:<8} {key[2]:<6} {len(g):>5} {m:>6}/{len(g):<3} {f_done:>6}/{f_tot:<4}")
    for ts, k in failed[:10]:
        print(f"FAILED {_tag(ts, k)}: {ts.dir}/.failed_{_task_name(k)}")


def reset_failed() -> None:
    n = 0
    for f in SEGRES_DIR.glob("*/.failed_*"):
        f.unlink()
        n += 1
    print(f"cleared {n} failure marker(s) in {SEGRES_DIR}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache-volumes", action="store_true", help="preprocess every scan once (CPU)")
    ap.add_argument("--procs", type=int, default=16, help="processes for --cache-volumes")
    ap.add_argument("--force", action="store_true", help="with --cache-volumes: rebuild")
    ap.add_argument("--regime", choices=["full", "few", "all"], default="all")
    ap.add_argument("--cohort", choices=list(COHORTS), default=None)
    ap.add_argument("--endpoint", choices=ENDPOINTS, default=None)
    ap.add_argument("--status", action="store_true", help="show progress and exit")
    ap.add_argument("--brief", action="store_true", help="with --status: one line")
    ap.add_argument("--reset-failed", action="store_true",
                    help="clear failure markers so failed networks are retried")
    a = ap.parse_args()
    regimes = ("full", "few") if a.regime == "all" else (a.regime,)
    cohorts = [a.cohort] if a.cohort else None
    endpoints = [a.endpoint] if a.endpoint else None
    if a.cache_volumes:
        cache_volumes(a.procs, a.force)
    elif a.status:
        status(regimes, cohorts, endpoints, a.brief)
    elif a.reset_failed:
        reset_failed()
    else:
        gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
        sys.exit(worker(regimes, cohorts, endpoints, gpu))
