"""IPCW Brier score / Integrated Brier Score, ported from the published pipeline.

Provenance: `working_space/generate_calibration_from_saved_model.py:503-562`
(`compute_brier_score_coxph_ipcw`), with byte-identical copies at
`generate_calibration_pytorch_head.py:389-438` and `calibration_all_models.py:529-591`.
This is the estimator that produced the published Table S6, and it is reused here
by explicit decision so the new A1 numbers sit alongside Table S6 unchanged.

Two properties of this estimator that differ from the textbook / sksurv version,
kept deliberately for continuity -- do NOT "fix" them without regenerating Table S6:
  * the Brier score is normalized by the SUM OF WEIGHTS, not by n;
  * subjects censored before t are dropped rather than down-weighted.

The one structural change from the original is a split into
`survival_probs_from_cox` (run once) and `ibs_from_probs` (run per bootstrap
replicate). The original recomputed survival functions on every call, which would
make 2000-replicate bootstrapping needlessly slow. The arithmetic is unchanged;
`verify_matches_original` below asserts equivalence.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter

# Verbatim from the published Config dataclasses
# (generate_calibration_from_saved_model.py:73).
DEFAULT_EVAL_TIMES = np.array([180, 365, 540, 730, 1095, 1460, 1825], dtype=float)


def eval_times_for(train_times: np.ndarray, train_events: np.ndarray,
                   base: Optional[np.ndarray] = None) -> np.ndarray:
    """Truncate the eval grid to 95% of training follow-up.

    Verbatim logic from generate_calibration_from_saved_model.py:764-767.
    """
    base = DEFAULT_EVAL_TIMES if base is None else base
    train_times = np.asarray(train_times, dtype=float)
    train_events = np.asarray(train_events)
    time_max = (train_times[train_events == 1].max()
                if train_events.sum() > 0 else train_times.max())
    out = base[base <= time_max * 0.95]
    if len(out) == 0:
        out = np.array([time_max * p for p in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)])
    return out


def survival_probs_from_cox(cph, df_test: pd.DataFrame, eval_times: np.ndarray) -> np.ndarray:
    """(n_patients, n_eval_times) predicted survival probabilities.

    Verbatim from generate_calibration_from_saved_model.py:512-517.
    """
    surv_funcs = cph.predict_survival_function(df_test)
    surv_probs = np.zeros((len(df_test), len(eval_times)))
    for i, (_, surv_func) in enumerate(surv_funcs.items()):
        surv_probs[i, :] = np.interp(eval_times, surv_func.index.values, surv_func.values)
    return surv_probs


def brier_scores_ipcw(surv_probs: np.ndarray, times: np.ndarray, events: np.ndarray,
                      eval_times: np.ndarray) -> np.ndarray:
    """Time-dependent IPCW Brier score at each eval time.

    Vectorized restatement of the published double loop. For each time t:
      * patients with an event at or before t contribute (S_hat - 0)^2, weighted
        by 1/G(t_i) -- the censoring survival at their own event time;
      * patients still at risk after t contribute (S_hat - 1)^2, weighted by 1/G(t);
      * weights are skipped when G <= 0.01, and the total is divided by the sum of
        the weights actually used.
    """
    times = np.asarray(times, dtype=float)
    events = np.asarray(events)

    kmf_censor = KaplanMeierFitter()
    kmf_censor.fit(times, 1 - events)

    # G at each patient's own event time (Case 1) -- computed once for all t.
    g_at_event = np.asarray(kmf_censor.predict(times), dtype=float)

    out = np.full(len(eval_times), np.nan)
    for j, t in enumerate(eval_times):
        g_at_t = float(kmf_censor.predict(t))
        s = surv_probs[:, j]

        case1 = (times <= t) & (events == 1) & (g_at_event > 0.01)
        case2 = (times > t) & (g_at_t > 0.01)

        w1 = 1.0 / g_at_event[case1]
        w2 = np.full(int(case2.sum()), 1.0 / g_at_t) if g_at_t > 0.01 else np.zeros(0)

        bs_sum = float(np.sum(w1 * (s[case1] - 0.0) ** 2) + np.sum(w2 * (s[case2] - 1.0) ** 2))
        weight_sum = float(w1.sum() + w2.sum())
        if weight_sum > 0:
            out[j] = bs_sum / weight_sum
    return out


def ibs_from_probs(surv_probs: np.ndarray, times: np.ndarray, events: np.ndarray,
                   eval_times: np.ndarray) -> float:
    """Integrated Brier Score by trapezoidal integration over the eval grid.

    Verbatim normalization from generate_calibration_from_saved_model.py:554-559.
    """
    bs = brier_scores_ipcw(surv_probs, times, events, eval_times)
    valid = ~np.isnan(bs)
    if valid.sum() < 2:
        return float("nan")
    et = np.asarray(eval_times, dtype=float)
    return float(np.trapz(bs[valid], et[valid]) / (et[valid][-1] - et[valid][0]))


def calibration_bins(surv_probs_at_t: np.ndarray, times: np.ndarray, events: np.ndarray,
                     time_point: float, n_bins: int = 5) -> Tuple[np.ndarray, np.ndarray, float]:
    """Quintile calibration: mean predicted vs Kaplan-Meier observed survival.

    Verbatim from generate_calibration_from_saved_model.py:597-651 plus the
    calibration-error line at :834. Bins with fewer than 3 patients are skipped.
    """
    pred_surv = np.asarray(surv_probs_at_t, dtype=float)
    times = np.asarray(times, dtype=float)
    events = np.asarray(events)

    bin_edges = np.unique(np.percentile(pred_surv, np.linspace(0, 100, n_bins + 1)))
    if len(bin_edges) < 3:
        bin_edges = np.linspace(pred_surv.min(), pred_surv.max(), n_bins + 1)

    predicted, observed = [], []
    for i in range(len(bin_edges) - 1):
        if i == len(bin_edges) - 2:
            mask = (pred_surv >= bin_edges[i]) & (pred_surv <= bin_edges[i + 1])
        else:
            mask = (pred_surv >= bin_edges[i]) & (pred_surv < bin_edges[i + 1])
        if mask.sum() < 3:
            continue
        predicted.append(pred_surv[mask].mean())
        kmf = KaplanMeierFitter()
        kmf.fit(times[mask], event_observed=events[mask])
        if time_point <= kmf.survival_function_.index.max():
            obs = kmf.survival_function_at_times(time_point).values[0]
        else:
            obs = kmf.survival_function_.iloc[-1].values[0]
        observed.append(obs)

    predicted, observed = np.asarray(predicted), np.asarray(observed)
    calib_error = float(np.mean(np.abs(predicted - observed))) if len(predicted) else float("nan")
    return predicted, observed, calib_error


def verify_matches_original(surv_probs: np.ndarray, times: np.ndarray, events: np.ndarray,
                            eval_times: np.ndarray, atol: float = 1e-10) -> bool:
    """Assert the vectorized Brier reproduces the published double loop exactly."""
    times = np.asarray(times, dtype=float)
    events = np.asarray(events)
    kmf_censor = KaplanMeierFitter()
    kmf_censor.fit(times, 1 - events)

    ref = np.zeros(len(eval_times))
    for j, t in enumerate(eval_times):                      # original loop, unmodified
        bs_sum, weight_sum = 0.0, 0.0
        for i in range(len(times)):
            pred_surv = surv_probs[i, j]
            if times[i] <= t and events[i] == 1:
                G = kmf_censor.predict(times[i])
                if G > 0.01:
                    w = 1.0 / G
                    bs_sum += w * (pred_surv - 0) ** 2
                    weight_sum += w
            elif times[i] > t:
                G = kmf_censor.predict(t)
                if G > 0.01:
                    w = 1.0 / G
                    bs_sum += w * (pred_surv - 1) ** 2
                    weight_sum += w
        ref[j] = bs_sum / weight_sum if weight_sum > 0 else np.nan

    got = brier_scores_ipcw(surv_probs, times, events, eval_times)
    both_nan = np.isnan(ref) & np.isnan(got)
    return bool(np.all(both_nan | np.isclose(ref, got, atol=atol, equal_nan=False)))
