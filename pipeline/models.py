"""The A1 model arms: clinical_only, the imaging model (ctfm_st, or segres_ft when
A1_IMAGING=segres_ft) and combined_A. combined_B is kept but no longer run.

Each `fit_*` returns a uniform `FitResult` so the driver can treat arms identically.
Risk scores are always oriented so that HIGHER = WORSE prognosis, matching
`bootstrap_ci.cindex`, which internally negates before calling `concordance_index`.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from ipcw_brier import survival_probs_from_cox

# Ridge grid used by the published CT-FM-ST arm (few_shot_single_institution_running.py:111-113).
RIDGE_GRID = [0.1, 1.0, 10.0]
# Applied to stage 2 of combined_A when events-per-variable falls below this.
MIN_EPV = 10.0
FALLBACK_PENALTY = 0.1


@dataclass
class FitResult:
    risk_test: np.ndarray
    surv_test: Optional[np.ndarray]
    n_params: int
    epv: float
    hyperparam: str
    loglik: Optional[float]
    penalized: bool
    model: object = None
    failed: str = ""
    # what a new patient's data must go through before `model` can score it (scalers, the
    # age standardization, ...): saved with the model so it can be reused on new patients
    preprocess: Optional[dict] = None


def _epv(train: pd.DataFrame, n_params: int) -> float:
    return float(train["event"].sum()) / max(n_params, 1)


def _cindex(times, events, risk) -> float:
    try:
        return concordance_index(times, -np.asarray(risk), events)
    except ZeroDivisionError:
        return float("nan")


def age_params(train: pd.DataFrame):
    """Mean and SD standardize_age uses -- saved with the clinical and combined models."""
    mu, sd = train["age"].mean(), train["age"].std()
    return float(mu), float(sd if sd and np.isfinite(sd) and sd > 0 else 1.0)


def standardize_age(train: pd.DataFrame, *others: pd.DataFrame):
    """z-score `age` on the training subsample; binary/ordinal covariates left raw."""
    mu, sd = age_params(train)
    out = []
    for df in (train,) + others:
        d = df.copy()
        d["age"] = (d["age"] - mu) / sd
        out.append(d)
    return out


def _fit_cox(df: pd.DataFrame, cols: Sequence[str], penalizer: float) -> CoxPHFitter:
    cph = CoxPHFitter(penalizer=penalizer, l1_ratio=0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cph.fit(df[list(cols) + ["time", "event"]], duration_col="time",
                event_col="event", show_progress=False)
    return cph


def _fit_cox_robust(df, cols, penalizer):
    """Fit, retrying with a small ridge if the unpenalized fit will not converge."""
    try:
        return _fit_cox(df, cols, penalizer), penalizer
    except Exception:
        if penalizer == 0.0:
            return _fit_cox(df, cols, FALLBACK_PENALTY), FALLBACK_PENALTY
        raise


# --------------------------------------------------------------------------- arms

def fit_clinical(train, test, cols: List[str], eval_times, penalizer: float = 0.0) -> FitResult:
    """Standard Cox on clinical covariates only.

    `penalizer` is normally supplied by `fit_clinical_and_combined`, so that this
    model and combined_A are always fitted with the same ridge penalty.
    """
    age_mu, age_sd = age_params(train)
    train, test = standardize_age(train, test)
    try:
        cph, pen = _fit_cox_robust(train, cols, penalizer)
    except Exception as e:
        return FitResult(np.full(len(test), np.nan), None, len(cols),
                         _epv(train, len(cols)), "failed", None, True, failed=str(e)[:200])
    return FitResult(
        risk_test=cph.predict_log_partial_hazard(test).values,
        surv_test=survival_probs_from_cox(cph, test, eval_times),
        n_params=len(cols), epv=_epv(train, len(cols)),
        hyperparam=f"penalizer={pen}", loglik=float(cph.log_likelihood_),
        penalized=pen > 0, model=cph,
        preprocess={"covariates": list(cols), "age_mean": age_mu, "age_sd": age_sd},
    )


def _feat_frame(feats: np.ndarray, df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(feats, columns=[f"f{i}" for i in range(feats.shape[1])])
    out["time"] = df["time"].values
    out["event"] = df["event"].values
    return out


def fit_ctfm_st(train, val, test, f_train, f_val, f_test, eval_times,
                fixed_lambda: Optional[float] = None) -> FitResult:
    """Imaging-only CT-FM-ST: frozen 512-d features -> z-score -> ridge Cox.

    Mirrors few_shot_single_institution_running.py:733-776, including selecting the
    ridge penalty on the FIXED validation split rather than fixing it at 1.0.
    """
    scaler = StandardScaler().fit(f_train)
    dtr = _feat_frame(scaler.transform(f_train), train)
    dva = _feat_frame(scaler.transform(f_val), val)
    dte = _feat_frame(scaler.transform(f_test), test)
    cols = [c for c in dtr.columns if c.startswith("f")]

    best, best_c, best_lam = None, -np.inf, None
    # `fixed_lambda` skips validation-based selection -- used when the validation split
    # is contained in the training set (external validation with --train-on all).
    for lam in ([fixed_lambda] if fixed_lambda is not None else RIDGE_GRID):
        try:
            cph = _fit_cox(dtr, cols, lam)
        except Exception:
            continue
        c = _cindex(dva["time"], dva["event"], cph.predict_log_partial_hazard(dva).values)
        if np.isfinite(c) and c > best_c:
            best, best_c, best_lam = cph, c, lam

    if best is None:
        return FitResult(np.full(len(test), np.nan), None, len(cols),
                         _epv(train, len(cols)), "failed", None, True,
                         failed="all ridge candidates failed")
    return FitResult(
        risk_test=best.predict_log_partial_hazard(dte).values,
        surv_test=survival_probs_from_cox(best, dte, eval_times),
        n_params=len(cols), epv=_epv(train, len(cols)),
        hyperparam=f"ridge={best_lam}", loglik=float(best.log_likelihood_),
        penalized=True, model=best,
        preprocess={"covariates": cols, "scaler_mean": scaler.mean_, "scaler_scale": scaler.scale_,
                    "note": "features -> (x - scaler_mean) / scaler_scale -> columns f0..f511"},
    )


def oof_splits(n: int, events, seed=42):
    """The stage-1 folds: leave-one-out below n=25, else 5-fold stratified on event.
    Shared with SegRes-FT (segres_cache.py), whose out-of-fold networks use the same folds."""
    if n < 25:
        return list(KFold(n_splits=n, shuffle=True, random_state=seed).split(np.arange(n)))
    return list(StratifiedKFold(n_splits=5, shuffle=True,
                                random_state=seed).split(np.arange(n), np.asarray(events)))


def _oof_imaging_scores(train, f_train, lam, seed=42) -> np.ndarray:
    """Out-of-fold stage-1 imaging risk scores for the training subsample.

    Without this, stage 2 sees an in-sample imaging score and the combined model looks
    spuriously good. Leave-one-out below n=25, else 5-fold stratified on event.
    """
    n = len(train)
    oof = np.full(n, np.nan)
    for tr_idx, va_idx in oof_splits(n, train["event"].values, seed):
        sub = train.iloc[tr_idx].reset_index(drop=True)
        if sub["event"].sum() < 2:          # too few events to fit this fold
            oof[va_idx] = 0.0               # -> "average risk" (see below)
            continue
        try:
            sc = StandardScaler().fit(f_train[tr_idx])
            cph = _fit_cox(_feat_frame(sc.transform(f_train[tr_idx]), sub),
                           [f"f{i}" for i in range(f_train.shape[1])], lam)
            dva = _feat_frame(sc.transform(f_train[va_idx]), train.iloc[va_idx])
            oof[va_idx] = cph.predict_log_partial_hazard(dva).values
        except Exception:
            oof[va_idx] = 0.0
    # A fold that cannot be fitted gives its patients a score of 0, i.e. average risk
    # (lifelines centres the linear predictor on the training mean). The alternative --
    # dropping those patients -- would train combined_A on fewer patients than
    # clinical_only. Imputing 0 is conservative: it can only dilute the imaging signal.
    return oof


def fit_combined_A(train, val, test, f_train, f_val, f_test, cols, eval_times,
                   imaging_fit: FitResult, seed=42,
                   penalizer: Optional[float] = None, stage1=None) -> FitResult:
    """Late fusion: clinical covariates + a single out-of-fold imaging risk score.

    `stage1` = (out-of-fold training scores, test scores, description) supplies the
    imaging score of a model that is not a ridge Cox on features (SegRes-FT, see
    fit_segres_ft). Default: CT-FM-ST, refitted out-of-fold here.
    """
    if stage1 is None:
        lam = float(imaging_fit.hyperparam.split("=")[1]) if "ridge=" in imaging_fit.hyperparam else 1.0
        oof = _oof_imaging_scores(train, f_train, lam, seed)
        test_score, s1 = imaging_fit.risk_test, f"stage1_ridge={lam}"
    else:
        oof, test_score, s1 = stage1
        oof, test_score = np.asarray(oof, float), np.asarray(test_score, float)
    ok = np.isfinite(oof)
    if ok.sum() < 5 or not np.isfinite(test_score).all():
        return FitResult(np.full(len(test), np.nan), None, len(cols) + 1,
                         _epv(train, len(cols) + 1), "failed", None, True,
                         failed=f"insufficient OOF coverage ({int(ok.sum())}/{len(train)})")

    # Standardize on the train OOF distribution; the same shift/scale is applied to test
    # so the hazard ratio reads per-SD of the imaging score.
    mu, sd = np.nanmean(oof), np.nanstd(oof)
    sd = sd if sd > 0 and np.isfinite(sd) else 1.0

    tr = train.loc[ok].copy()
    tr["img_score"] = (oof[ok] - mu) / sd
    te = test.copy()
    te["img_score"] = (test_score - mu) / sd
    age_mu, age_sd = age_params(tr)
    tr, te = standardize_age(tr, te)

    cols2 = list(cols) + ["img_score"]
    epv = _epv(tr, len(cols2))
    pen = penalizer if penalizer is not None else (0.0 if epv >= MIN_EPV else FALLBACK_PENALTY)
    try:
        cph, pen = _fit_cox_robust(tr, cols2, pen)
    except Exception as e:
        return FitResult(np.full(len(test), np.nan), None, len(cols2), epv,
                         "failed", None, True, failed=str(e)[:200])
    return FitResult(
        risk_test=cph.predict_log_partial_hazard(te).values,
        surv_test=survival_probs_from_cox(cph, te, eval_times),
        n_params=len(cols2), epv=epv,
        hyperparam=f"penalizer={pen};{s1}",
        loglik=float(cph.log_likelihood_), penalized=pen > 0, model=cph,
        preprocess={"covariates": cols2, "age_mean": age_mu, "age_sd": age_sd,
                    "img_score_mean": float(mu), "img_score_sd": float(sd),
                    "note": "img_score = (imaging model's risk - img_score_mean) / img_score_sd; "
                            "the imaging model is saved beside this one"},
    )


def dl_score_wald(comb: FitResult):
    """Wald test of the DL score inside the fitted combined_A model (training data).

    Returns (HR per SD, lo95, hi95, p, penalized) or None. Two caveats, reported with
    the number rather than hidden: the p-value is not valid if the fit was penalized,
    and it is optimistic because the DL score was itself estimated from data (stage 1).
    """
    m = comb.model if comb is not None else None
    if comb is None or comb.failed or m is None or "img_score" not in getattr(m, "params_", {}):
        return None
    s = m.summary.loc["img_score"]
    return (float(s["exp(coef)"]), float(s["exp(coef) lower 95%"]),
            float(s["exp(coef) upper 95%"]), float(s["p"]), bool(comb.penalized))


def dl_score_independent_test(times, events, clin_risk, img_risk):
    """Is the DL score significant in patients NEITHER score was fitted on?

    On held-out / external patients, fit  Cox(time, event ~ clinical_score + dl_score),
    where both scores are frozen predictions from the training cohort. Nothing about
    the scores is estimated here, so -- unlike the training-set Wald test -- the
    p-value is not inflated by stage-1 estimation. Scores are z-scored on this set, so
    the HR reads per SD (the p-value is unaffected by the scaling).
    Returns (HR per SD, lo95, hi95, p) or None.
    """
    c, g = np.asarray(clin_risk, float), np.asarray(img_risk, float)
    if not (np.isfinite(c).all() and np.isfinite(g).all()) or c.std() == 0 or g.std() == 0:
        return None
    df = pd.DataFrame({"time": np.asarray(times, float), "event": np.asarray(events, int),
                       "clinical_score": (c - c.mean()) / c.std(),
                       "dl_score": (g - g.mean()) / g.std()})
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cph = CoxPHFitter().fit(df, "time", "event")
    except Exception:
        return None
    s = cph.summary.loc["dl_score"]
    return (float(s["exp(coef)"]), float(s["exp(coef) lower 95%"]),
            float(s["exp(coef) upper 95%"]), float(s["p"]))


def _used_penalty(res: FitResult):
    import re
    m = re.search(r"penalizer=([\d.]+)", res.hyperparam or "")
    return float(m.group(1)) if m else None


def fit_clinical_and_combined(train, val, test, f_train, f_val, f_test, cols, eval_times,
                              imaging_fit: FitResult, seed=42, stage1=None):
    """Fit clinical_only and combined_A under IDENTICAL training settings.

    The two models must differ only by the imaging_score column, so they share one
    ridge-penalty decision. It is taken from the larger model (combined_A, one extra
    covariate), whose events-per-variable is the binding constraint. If either fit
    needs the fallback penalty to converge, both are refitted with it. Both also train
    on exactly the same patients (see `_oof_imaging_scores`). `stage1`: see fit_combined_A.
    """
    epv = _epv(train, len(cols) + 1)
    pen = 0.0 if epv >= MIN_EPV else FALLBACK_PENALTY
    clin = fit_clinical(train, test, cols, eval_times, penalizer=pen)
    comb = fit_combined_A(train, val, test, f_train, f_val, f_test, cols, eval_times,
                          imaging_fit, seed, penalizer=pen, stage1=stage1)
    p_clin, p_comb = _used_penalty(clin), _used_penalty(comb)
    if None not in (p_clin, p_comb) and p_clin != p_comb:
        pen = max(p_clin, p_comb)
        clin = fit_clinical(train, test, cols, eval_times, penalizer=pen)
        comb = fit_combined_A(train, val, test, f_train, f_val, f_test, cols, eval_times,
                              imaging_fit, seed, penalizer=pen, stage1=stage1)
    return clin, comb


def fit_segres_ft(train, val, test, eval_times, cohort, endpoint, seed=42,
                  test_cohort=None):
    """Imaging-only SegRes-FT, read from the networks train_segres.py trained.

    Nothing is fitted here: the network trained on exactly these training patients (with
    early stopping on exactly this validation split) supplies the test risk scores. Like
    lifelines' Cox, the scores are centred on the training patients' mean, and survival
    curves use the Breslow baseline hazard of the training patients.

    Returns (FitResult, stage1). stage1 is the combined model's imaging score: for the
    training patients, the out-of-fold networks' outputs (same folds as CT-FM-ST's
    stage 1); for the test patients, the main network's. Every network's output is
    first z-scored on its OWN training patients, so outputs of separately trained
    networks share one scale -- a network's offset and scale are arbitrary, because
    the Cox loss ignores the offset.
    """
    import segres_cache as SC                    # numpy only; no torch in analysis jobs
    e = SC.load_entry(cohort, endpoint, train, val, seed)
    r_tr = e.raw(cohort, train["key"].values)
    r_te = e.raw(test_cohort or cohort, test["key"].values)
    mu, sd = float(r_tr.mean()), float(r_tr.std())
    sd = sd if sd > 0 and np.isfinite(sd) else 1.0
    fit = FitResult(
        risk_test=r_te - mu,
        surv_test=SC.breslow_survival(r_tr - mu, train["time"].values, train["event"].values,
                                      r_te - mu, eval_times),
        n_params=e.n_params, epv=_epv(train, e.n_params),
        hyperparam=f"epochs={e.best_epoch}/{e.epochs_run};val_c={e.best_val_c:.3f}",
        loglik=None, penalized=True, model=None,
    )
    stage1 = (SC.oof_scores(e, train, seed), (r_te - mu) / sd,
              f"stage1=segres_ft_oof{e.n_folds}")
    return fit, stage1


def fit_combined_B(train, val, test, f_train, f_val, f_test, cols, eval_times) -> FitResult:
    """Early fusion: clinical covariates + all 512 imaging features in one Cox.

    Uses sksurv's elastic-net Cox with `penalty_factor=0` on the clinical block, so
    established predictors (HPV, T, N, ...) enter unpenalized while the 512 imaging
    features are shrunk. A single global ridge would otherwise shrink HPV status exactly
    as hard as imaging feature #387. Alpha is selected on the fixed validation split.
    """
    from sksurv.linear_model import CoxnetSurvivalAnalysis
    from sksurv.util import Surv

    train, val, test = standardize_age(train, val, test)
    scaler = StandardScaler().fit(f_train)

    def _X(df, feats):
        return np.hstack([df[cols].to_numpy(dtype=float), scaler.transform(feats)])

    X_tr, X_va, X_te = _X(train, f_train), _X(val, f_val), _X(test, f_test)
    y_tr = Surv.from_arrays(event=train["event"].to_numpy(bool),
                            time=train["time"].to_numpy(float))
    n_params = X_tr.shape[1]
    epv = _epv(train, n_params)
    pf = np.r_[np.zeros(len(cols)), np.ones(f_train.shape[1])]

    def _fail(msg):
        return FitResult(np.full(len(test), np.nan), None, n_params, epv,
                         "failed", None, True, failed=msg[:200])

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = CoxnetSurvivalAnalysis(l1_ratio=0.9, penalty_factor=pf,
                                           fit_baseline_model=True, max_iter=200_000)
            model.fit(X_tr, y_tr)
    except Exception as e:
        return _fail(f"coxnet fit failed: {e}")

    best_a, best_c = None, -np.inf
    for a in model.alphas_:
        try:
            c = _cindex(val["time"].values, val["event"].values, model.predict(X_va, alpha=a))
        except Exception:
            continue
        if np.isfinite(c) and c > best_c:
            best_a, best_c = float(a), c
    if best_a is None:
        return _fail("no alpha produced a usable validation C-index")

    try:
        risk = model.predict(X_te, alpha=best_a)
        fns = model.predict_survival_function(X_te, alpha=best_a)
        surv = np.vstack([np.interp(eval_times, fn.x, fn.y) for fn in fns])
    except Exception as e:
        return _fail(f"coxnet prediction failed: {e}")

    return FitResult(
        risk_test=np.asarray(risk, dtype=float), surv_test=surv,
        n_params=n_params, epv=epv,
        hyperparam=f"alpha={best_a:.5g};l1_ratio=0.9",
        loglik=None,            # elastic-net objective is not a partial likelihood -> no LRT
        penalized=True, model=model,
    )
