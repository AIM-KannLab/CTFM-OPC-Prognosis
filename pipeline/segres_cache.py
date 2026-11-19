"""SegRes-FT network store: what train_segres.py writes and the analysis reads back.

numpy only, so the analysis jobs (run_a1.py, run_external.py) never import torch.

Every network is identified by its TRAINING SET -- cohort, endpoint, training size and
seed -- exactly as run_a1.py / run_external.py form it, so an analysis can only ever
read the network that was trained on its own training patients:

    <SEGRES_DIR>/<cohort>_<endpoint>_n<size>_s<seed>/
        main.json        training record: fingerprint of the training + validation
                         patients and labels, epochs, validation C-index per epoch
        main.npz         network output (log-hazard) for every patient it scored
        main.pt          weights (full-shot networks only)
        fold<k>of<K>.npz out-of-fold network k: outputs for its held-out patients, and
                         mean / SD of its outputs on its own training patients
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from cohort import (COHORTS, ENDPOINTS, SEGRES_DIR, SEGRES_VARIANT, build_cohort,
                    split_dataset_ratio)
from models import oof_splits
from subsample import SEEDS, shot_grid, stratified_subsample

FULL_SHOT_SEED = 42           # run_a1.py / run_external.py fit the full-shot models with seed 42
# SegRes-E2E is an imaging-only baseline (run_baselines.py): no combined model, so no
# out-of-fold networks, and -- like every stochastic model in the submitted Table S4 --
# 5 seeds at full shot as well as at every few-shot size.
IMAGING_ONLY = SEGRES_VARIANT == "e2e"
FULL_SHOT_SEEDS = SEEDS if IMAGING_ONLY else [FULL_SHOT_SEED]


class MissingNetworks(RuntimeError):
    pass


def fingerprint(train: pd.DataFrame, val: pd.DataFrame) -> str:
    """md5 of the training and validation patients, in order, with their labels."""
    h = hashlib.md5()
    for d in (train, val):
        h.update("\n".join(map(str, d["key"])).encode())
        h.update(np.asarray(d["time"], np.float64).tobytes())
        h.update(np.asarray(d["event"], np.int64).tobytes())
        h.update(b"|")
    return h.hexdigest()


def entry_dir(cohort: str, endpoint: str, n: int, seed: int) -> Path:
    return SEGRES_DIR / f"{cohort}_{endpoint}_n{n}_s{seed}"


@dataclass
class TrainingSet:
    """One network's training problem, formed exactly as run_a1.py forms it."""
    cohort: str
    endpoint: str
    regime: str               # "full" or "few"
    seed: int
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame        # the cohort's held-out test split
    other: str | None         # full shot: the other cohort, scored for external validation

    @property
    def dir(self) -> Path:
        return entry_dir(self.cohort, self.endpoint, len(self.train), self.seed)

    @property
    def folds(self):
        if IMAGING_ONLY:
            return []
        return oof_splits(len(self.train), self.train["event"].values, self.seed)


def training_sets(cohorts=None, endpoints=None, regimes=("full", "few")):
    """Every training set the A1 study and the external validation need.

    Full shot: the whole training split, seed 42 -- shared by run_a1.py (internal test)
    and run_external.py (the same split of the source cohort), so its network also
    scores every patient of the other cohort. Few shot: the shot grid x SEEDS.
    For SegRes-E2E (imaging only) full shot is trained with every seed in SEEDS.
    """
    for co in cohorts or list(COHORTS):
        other = next(c for c in COHORTS if c != co)
        for ep in endpoints or ENDPOINTS:
            df = build_cohort(co, ep)
            tr_full, va, te = split_dataset_ratio(df)
            if "full" in regimes:
                for seed in FULL_SHOT_SEEDS:
                    yield TrainingSet(co, ep, "full", seed, tr_full, va, te, other)
            if "few" in regimes:
                for shot in [s for s in shot_grid(len(tr_full)) if s < len(tr_full)]:
                    for seed in SEEDS:
                        sub = stratified_subsample(tr_full, shot, seed)
                        yield TrainingSet(co, ep, "few", seed, sub, va, te, None)


def check_trained(cohorts, endpoints, regimes) -> None:
    """Stop early, before any fitting, unless every network these cells need is on disk."""
    missing = []
    for ts in training_sets(cohorts, endpoints, regimes):
        K = len(ts.folds)
        if not (ts.dir / "main.json").exists() or any(
                not (ts.dir / f"fold{k}of{K}.npz").exists() for k in range(K)):
            missing.append(ts.dir.name)
    if missing:
        regime = "all" if set(regimes) == {"full", "few"} else next(iter(regimes))
        raise MissingNetworks(
            f"{len(missing)} SegRes-FT training sets are not fully trained in {SEGRES_DIR} "
            f"(e.g. {missing[0]}) -- train them first, one worker per GPU:\n"
            f"   CUDA_VISIBLE_DEVICES=<gpu> python train_segres.py --regime {regime}")


class Entry:
    """A trained network (main + out-of-fold networks) read back from disk."""

    def __init__(self, d: Path):
        self.dir = d
        self.meta = json.loads((d / "main.json").read_text())
        z = np.load(d / "main.npz", allow_pickle=False)
        self._pred = {}
        for c in COHORTS:
            if f"keys_{c}" in z.files:
                self._pred[c] = dict(zip(z[f"keys_{c}"].tolist(), z[f"risk_{c}"]))
        self.n_params = int(self.meta["n_params"])
        self.best_epoch = int(self.meta["best_epoch"])
        self.epochs_run = int(self.meta["epochs_run"])
        self.best_val_c = float(self.meta["best_val_c"])
        self.n_folds = int(self.meta["n_folds"])

    def raw(self, cohort: str, keys) -> np.ndarray:
        """Network output (log-hazard, arbitrary offset) for these patients."""
        if cohort not in self._pred:
            raise MissingNetworks(f"{self.dir.name} did not score {cohort} patients")
        p = self._pred[cohort]
        missing = [k for k in keys if k not in p]
        if missing:
            raise MissingNetworks(f"{self.dir.name}: no output for {len(missing)} {cohort} "
                                  f"patients (e.g. {missing[0]})")
        return np.asarray([p[k] for k in keys], dtype=float)


def load_entry(cohort, endpoint, train, val, seed) -> Entry:
    d = entry_dir(cohort, endpoint, len(train), seed)
    if not (d / "main.json").exists():
        raise MissingNetworks(f"no trained network {d.name} in {SEGRES_DIR} -- "
                              f"train it first: python train_segres.py (see its docstring)")
    e = Entry(d)
    if e.meta["fingerprint"] != fingerprint(train, val):
        raise ValueError(f"{d.name} was trained on different patients or labels than this "
                         f"analysis uses (data version or split changed?) -- refusing to use it")
    missing = [k for k in range(e.n_folds) if not (d / f"fold{k}of{e.n_folds}.npz").exists()]
    if missing:
        raise MissingNetworks(f"{d.name}: out-of-fold networks {missing} not trained yet")
    return e


def oof_scores(e: Entry, train: pd.DataFrame, seed) -> np.ndarray:
    """Out-of-fold imaging score of every training patient, aligned with `train`.

    Each fold network's outputs are z-scored on its own training patients. A fold that
    could not be trained (fewer than 2 events) or that diverged gives its patients 0 =
    average risk -- the rule CT-FM-ST's stage 1 uses (models._oof_imaging_scores).
    """
    keys = train["key"].values
    folds = oof_splits(len(train), train["event"].values, seed)
    if len(folds) != e.n_folds:
        raise ValueError(f"{e.dir.name}: {e.n_folds} fold networks on disk, {len(folds)} folds expected")
    oof = np.full(len(train), np.nan)
    for k, (_, va_idx) in enumerate(folds):
        z = np.load(e.dir / f"fold{k}of{e.n_folds}.npz", allow_pickle=False)
        if not np.array_equal(z["keys"], keys[va_idx]):
            raise ValueError(f"{e.dir.name} fold {k}: held-out patients differ from the expected fold")
        r, mu, sd = z["risk"].astype(float), float(z["fit_mean"]), float(z["fit_sd"])
        ok = (not bool(z["failed"])) and np.isfinite(r).all() and np.isfinite(mu) and sd > 0
        oof[va_idx] = (r - mu) / sd if ok else 0.0
    if not np.isfinite(oof).all():
        raise ValueError(f"{e.dir.name}: some training patients have no out-of-fold score")
    return oof


def breslow_survival(train_risk, train_time, train_event, test_risk, eval_times) -> np.ndarray:
    """(n_test, n_eval_times) survival probabilities, S(t|x) = exp(-H0(t) exp(r(x))).

    H0 = Breslow baseline cumulative hazard of the training patients, given their risk
    scores. Like lifelines' predict_survival_function + ipcw_brier.survival_probs_from_cox,
    S is evaluated at the training durations and linearly interpolated at `eval_times`.
    """
    r = np.asarray(train_risk, float)
    t = np.asarray(train_time, float)
    e = np.asarray(train_event).astype(bool)
    shift = r.max()                                   # numerical stability; cancels below
    w = np.exp(r - shift)
    grid = np.unique(t)
    at_risk = np.array([w[t >= g].sum() for g in grid])
    d = np.array([(e & (t == g)).sum() for g in grid])
    h0 = np.cumsum(np.where(at_risk > 0, d / np.where(at_risk > 0, at_risk, 1), 0.0))
    rel = np.exp(np.asarray(test_risk, float) - shift)
    surv = np.exp(-np.outer(rel, h0))                 # (n_test, len(grid))
    return np.vstack([np.interp(eval_times, grid, s) for s in surv])
