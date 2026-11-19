"""The submitted study's calibration / IBS functions, for refit_segres_lrf_dm.py.

Copied VERBATIM from the submitted study's generate_calibration_pytorch_head.py (the script that
produced the submitted Table S6 / Figure 5): Config.eval_times, breslow_baseline_survival,
risk_to_survival_probs and compute_brier_score_ipcw. `km_bins` is copied verbatim from
optional/score_paper_networks_ibs.py. The revision code imported these from the original script
(via score_paper_networks_ibs.py); they are copied here so that the refit step does not need the
original code base. Nothing in the arithmetic is changed.

Also re-exported, as score_paper_networks_ibs.py did: bootstrap_ci (bootstrap_ci.py) and
ibs_from_probs (ipcw_brier.py). The module can be used in place of both the original script
(`G`) and score_paper_networks_ibs (`SP`):  `import submitted_calibration as SP; G = SP.G`.
"""
from __future__ import annotations

import logging
import sys
import warnings
from dataclasses import dataclass, field
from typing import Tuple

warnings.filterwarnings('ignore')          # as importing the original script did

import numpy as np                          # noqa: E402
from lifelines import KaplanMeierFitter     # noqa: E402

from bootstrap_ci import bootstrap_ci       # noqa: E402,F401  (re-exported)
from ipcw_brier import ibs_from_probs       # noqa: E402,F401  (re-exported)

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)            # score_paper_networks_ibs.py: "it logs every Brier computation"


@dataclass
class Config:
    """The one field of the original Config these functions' callers use."""
    # Fixed clinical evaluation times (same as calibration_all_models.py)
    eval_times: np.ndarray = field(default_factory=lambda: np.array([180, 365, 540, 730, 1095, 1460, 1825]))


# =============================================================================
# Breslow Baseline Estimator
# =============================================================================
def breslow_baseline_survival(times: np.ndarray, events: np.ndarray,
                               risk_scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Estimate baseline survival S0(t) using Breslow estimator with centered risk scores.

    Centering prevents numerical overflow in exp() and is mathematically equivalent
    (the mean is absorbed into the baseline).
    """
    risk_center = float(np.mean(risk_scores))
    centered_risks = risk_scores - risk_center

    order = np.argsort(times)
    sorted_times = times[order]
    sorted_events = events[order]
    sorted_risks = centered_risks[order]

    event_times = sorted_times[sorted_events == 1]
    unique_event_times = np.unique(event_times)

    if len(unique_event_times) == 0:
        return np.array([times.max()]), np.array([1.0]), risk_center

    exp_risks = np.exp(np.clip(sorted_risks, -500, 500))

    cumulative_hazard = 0.0
    baseline_hazards = []

    for t in unique_event_times:
        d_t = np.sum((sorted_times == t) & (sorted_events == 1))
        at_risk_mask = sorted_times >= t
        risk_sum = np.sum(exp_risks[at_risk_mask])

        if risk_sum > 1e-300:
            cumulative_hazard += d_t / risk_sum

        baseline_hazards.append(cumulative_hazard)

    baseline_survival = np.exp(-np.array(baseline_hazards))
    return unique_event_times, baseline_survival, risk_center


def risk_to_survival_probs(test_risk_scores: np.ndarray, baseline_times: np.ndarray,
                           baseline_survival: np.ndarray, eval_times: np.ndarray,
                           risk_center: float) -> np.ndarray:
    """Convert risk scores to survival probabilities: S(t|x) = S0(t)^exp(centered_risk)"""
    n_test = len(test_risk_scores)
    n_times = len(eval_times)
    surv_probs = np.zeros((n_test, n_times))

    centered_test = test_risk_scores - risk_center
    exp_risks = np.exp(np.clip(centered_test, -500, 500))

    for j, t in enumerate(eval_times):
        if t <= baseline_times[0]:
            S0_t = 1.0
        elif t >= baseline_times[-1]:
            S0_t = baseline_survival[-1]
        else:
            idx = np.searchsorted(baseline_times, t, side='right') - 1
            S0_t = baseline_survival[max(0, idx)]

        if S0_t <= 0:
            surv_probs[:, j] = 0.0
        elif S0_t >= 1.0:
            surv_probs[:, j] = 1.0
        else:
            log_S0 = np.log(S0_t)
            log_surv = log_S0 * exp_risks
            surv_probs[:, j] = np.clip(np.exp(log_surv), 0, 1)

    return surv_probs


# =============================================================================
# Brier Score with custom IPCW (same as calibration_all_models.py)
# =============================================================================
def compute_brier_score_ipcw(surv_probs: np.ndarray, eval_times: np.ndarray,
                             test_times: np.ndarray, test_events: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Compute time-dependent Brier score with IPCW (same as calibration_all_models.py)."""
    n_samples = len(test_times)
    n_times = len(eval_times)

    # Fit censoring distribution
    kmf_censor = KaplanMeierFitter()
    kmf_censor.fit(test_times, 1 - test_events)

    brier_scores = np.zeros(n_times)

    for j, t in enumerate(eval_times):
        bs_sum, weight_sum = 0.0, 0.0

        for i in range(n_samples):
            pred_surv = surv_probs[i, j]

            # Case 1: Event occurred before time t
            if test_times[i] <= t and test_events[i] == 1:
                actual = 0
                G = kmf_censor.predict(test_times[i])
                if G > 0.01:
                    weight = 1.0 / G
                    bs_sum += weight * (pred_surv - actual) ** 2
                    weight_sum += weight

            # Case 2: Still at risk at time t
            elif test_times[i] > t:
                actual = 1
                G = kmf_censor.predict(t)
                if G > 0.01:
                    weight = 1.0 / G
                    bs_sum += weight * (pred_surv - actual) ** 2
                    weight_sum += weight

        if weight_sum > 0:
            brier_scores[j] = bs_sum / weight_sum
        else:
            brier_scores[j] = np.nan

    # Integrated Brier Score
    valid = ~np.isnan(brier_scores)
    if valid.sum() >= 2:
        ibs = np.trapz(brier_scores[valid], eval_times[valid]) / (eval_times[valid][-1] - eval_times[valid][0])
    else:
        ibs = np.nan

    logger.info(f"  Brier scores: {brier_scores}")
    return eval_times, brier_scores, ibs


# =============================================================================
# Calibration bins (verbatim from score_paper_networks_ibs.py)
# =============================================================================
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


G = sys.modules[__name__]                   # this module stands in for the original script
