"""Bootstrap 95% CIs and paired model comparisons for C-index and IBS.

Addresses reviewer comments R3-7 and R3-11 (see revision_coverage_map.md):
bootstrapped CIs for C-index and IBS, plus paired comparisons between models.

Key design point: all models are evaluated on the SAME bootstrap resamples, so
differences are paired. Comparing two independent CIs for overlap is a weaker
and more conservative test than testing the CI of the difference directly.
"""
import numpy as np
import pandas as pd
from lifelines.utils import concordance_index
from sksurv.metrics import integrated_brier_score
from sksurv.util import Surv


def _resample_idx(rng, events, stratify):
    """Bootstrap indices, optionally stratified on event status.

    Stratification keeps the number of events stable across replicates, which
    matters for small test sets (Maastro n=26) where an unstratified resample
    can land on very few events and make the C-index undefined or unstable.
    """
    n = len(events)
    if not stratify:
        return rng.integers(0, n, n)
    idx = []
    for val in (0, 1):
        pool = np.flatnonzero(events == val)
        if len(pool):
            idx.append(rng.choice(pool, len(pool), replace=True))
    return np.concatenate(idx)


def cindex(times, events, risk):
    """C-index. `risk` is a risk score: higher = worse prognosis."""
    return concordance_index(times, -np.asarray(risk), events)


def ibs(times, events, surv_prob, eval_times, train_times=None, train_events=None):
    """Integrated Brier Score.

    surv_prob : (n_patients, n_eval_times) predicted survival probabilities.
    Censoring distribution is estimated from the training set when supplied,
    which is the correct IPCW reference; otherwise from the test set itself.
    """
    test_y = Surv.from_arrays(event=events.astype(bool), time=times)
    train_y = (test_y if train_times is None
               else Surv.from_arrays(event=train_events.astype(bool), time=train_times))
    return integrated_brier_score(train_y, test_y, surv_prob, eval_times)


def bootstrap_ci(metric_fn, times, events, B=2000, seed=42, stratify=True, alpha=0.05):
    """Percentile bootstrap CI for a metric evaluated on the test set.

    metric_fn(idx) -> float, where idx indexes into the test set.
    Returns (point_estimate, lo, hi, n_valid_replicates).
    """
    times, events = np.asarray(times), np.asarray(events)
    rng = np.random.default_rng(seed)
    point = metric_fn(np.arange(len(times)))
    vals = []
    for _ in range(B):
        idx = _resample_idx(rng, events, stratify)
        if events[idx].sum() < 2:          # need >=2 events for a usable estimate
            continue
        try:
            vals.append(metric_fn(idx))
        except (ZeroDivisionError, ValueError):
            continue
    vals = np.asarray(vals)
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, lo, hi, len(vals)


def paired_bootstrap(metric_fn_a, metric_fn_b, times, events,
                     B=2000, seed=42, stratify=True, alpha=0.05):
    """Paired bootstrap for the DIFFERENCE between two models (a - b).

    Both models are scored on identical resamples. Returns
    (delta_point, lo, hi, p_two_sided).

    The p-value is the standard two-sided bootstrap proportion: the fraction of
    replicates on the wrong side of zero, doubled. With B=2000 the smallest
    reportable value is p<0.001, so report it as such rather than as 0.
    """
    times, events = np.asarray(times), np.asarray(events)
    rng = np.random.default_rng(seed)
    full = np.arange(len(times))
    delta_point = metric_fn_a(full) - metric_fn_b(full)
    deltas = []
    for _ in range(B):
        idx = _resample_idx(rng, events, stratify)
        if events[idx].sum() < 2:
            continue
        try:
            deltas.append(metric_fn_a(idx) - metric_fn_b(idx))
        except (ZeroDivisionError, ValueError):
            continue
    deltas = np.asarray(deltas)
    lo, hi = np.percentile(deltas, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    p = 2 * min((deltas <= 0).mean(), (deltas >= 0).mean())
    return delta_point, lo, hi, min(max(p, 1 / (len(deltas) + 1)), 1.0)


def fmt(point, lo, hi, dp=3):
    """Format as the journal expects: 0.715 (95% CI 0.672-0.758)."""
    return f"{point:.{dp}f} (95% CI {lo:.{dp}f}–{hi:.{dp}f})"
