"""Kaplan-Meier figures -- reviewer item A5 (R2-5, R2-6, R2-15), with numbers at risk.

Two figures, answering two different requests:

  cohort   What the outcome data look like, per endpoint and cohort. No model involved.
           This is the figure R2 asked for to settle the event-count question ("KM plots
           of the data would be helpful here"). OS and RFS are Kaplan-Meier survival
           curves. LR and DM are CUMULATIVE INCIDENCE curves with death as a competing
           risk (Aalen-Johansen), not 1 - KM: death is censored in those endpoints, and
           with 382 deaths in Radcure a Kaplan-Meier estimate would overstate how many
           patients actually recur. This is also the competing-risk handling R3 asked
           about -- the estimator matches how the endpoints are defined.

  tertile  Whether a model separates risk: the held-out test set split into low / medium
           / high thirds of predicted risk, with a log-rank p (R2-6). FULL SHOT ONLY --
           a few-shot size has five models, one per seed, so there is no single risk
           score to cut into thirds, and pooling the seeds would place each patient in
           the plot five times with conflicting labels and an invalid log-rank p.
           The external (Radcure -> Maastro) version of this figure is made by
           make_external_report.py; this one covers the internal test sets.

Every panel carries a numbers-at-risk row, which the submitted figures lack.

    python make_km.py --runs results_corrected_hu_n8_<id>      # both figures
    python make_km.py                                          # cohort figure only
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import AalenJohansenFitter, KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test

from cohort import COHORTS, ENDPOINT_PAPER_NAME, ENDPOINTS, build_cohort
from make_report import INK, INK2, MUTED, SURFACE

DAYS = 365.25
HORIZON = 5                      # years shown
TICKS = list(range(0, HORIZON + 1))
# cohort figure: the two cohorts, categorical slots 1 and 2 (validated pair)
COHORT_COLOR = {"Radcure": "#2a78d6", "Maastro": "#eb6834"}
# tertile figure: one hue, light -> dark, so the ordering reads as an ordering
TERTILE_COLOR = {0: "#86b6ef", 1: "#2a78d6", 2: "#104281"}
TERTILE_NAME = {0: "low", 1: "medium", 2: "high"}
SURVIVAL_ENDPOINTS = {"os", "efs"}       # KM; the rest get cumulative incidence


def _at_risk(ax, rows, ticks=TICKS, label_w=22.0, xlabel=None):
    """Numbers-at-risk rows under the axis. `rows` = [(label, colour, times_in_years)].

    Returns the lowest axes-fraction y it used. With `xlabel` the axis label is drawn
    below that, instead of by set_xlabel, which would otherwise land on the numbers.
    """
    y0 = -0.17
    for k, (name, col, t) in enumerate(rows):
        y = y0 - 0.085 * k
        ax.annotate(name, (0, y), xycoords=("axes fraction", "axes fraction"),
                    xytext=(-label_w, 0), textcoords="offset points",
                    fontsize=6.8, color=col, ha="right", va="center", annotation_clip=False)
        for x in ticks:
            ax.annotate(f"{int((np.asarray(t) >= x).sum())}", (x, y),
                        xycoords=("data", "axes fraction"), fontsize=6.8, color=INK2,
                        ha="center", va="center", annotation_clip=False)
    ax.annotate("at risk", (0, y0 + 0.075), xycoords=("axes fraction", "axes fraction"),
                xytext=(-label_w, 0), textcoords="offset points", fontsize=6.8,
                color=MUTED, ha="right", va="center", annotation_clip=False)
    low = y0 - 0.085 * max(len(rows) - 1, 0)
    if xlabel:
        ax.annotate(xlabel, (0.5, low - 0.1), xycoords="axes fraction", fontsize=9,
                    color=INK2, ha="center", va="center", annotation_clip=False)
    return low


def _competing_codes(cohort: str, endpoint: str):
    """(time in years, 0 censored / 1 failure / 2 died before any failure) for LR and DM.

    Death is censored in these endpoints, so a patient who dies without the failure can
    never have it: that is the competing event. Verified when building: such patients'
    follow-up time equals their time of death.
    """
    d = build_cohort(cohort, endpoint)
    os_ = build_cohort(cohort, "os").set_index("key").loc[d["key"]]
    code = np.where(d["event"].values == 1, 1,
                    np.where(os_["event"].values == 1, 2, 0))
    return d["time"].values / DAYS, code


def fig_cohort(out: Path):
    """Outcome curves per endpoint, both cohorts. No model."""
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 9.6))
    for ax, ep in zip(axes.flat, ENDPOINTS):
        survival = ep in SURVIVAL_ENDPOINTS
        rows, top = [], 0.0
        for co in COHORTS:
            col = COHORT_COLOR[co]
            if survival:
                d = build_cohort(co, ep)
                t, e = d["time"].values / DAYS, d["event"].values
                km = KaplanMeierFitter().fit(t, e)
                x = km.survival_function_.index.values
                y = km.survival_function_.iloc[:, 0].values
                keep = x <= HORIZON
                x, y = np.r_[x[keep], HORIZON], np.r_[y[keep], y[keep][-1]]
                n_ev = int(e.sum())
            else:
                t, code = _competing_codes(co, ep)
                with warnings.catch_warnings():       # ties are jittered internally
                    warnings.simplefilter("ignore")
                    aj = AalenJohansenFitter(seed=42).fit(t, code, event_of_interest=1)
                x = aj.cumulative_density_.index.values
                y = aj.cumulative_density_.iloc[:, 0].values
                keep = x <= HORIZON
                x, y = np.r_[x[keep], HORIZON], np.r_[y[keep], y[keep][-1]]
                n_ev = int((code == 1).sum())
            ax.step(x, y, where="post", color=col, lw=2, zorder=3,
                    label=f"{co} ({n_ev}/{len(t)})")
            rows.append((co, col, t))
            top = max(top, float(y.max()))
        ax.set_xlim(0, HORIZON)
        # survival curves keep the full 0-1 scale; incidence is scaled to the data, so the
        # curves are readable -- the axis label and the tick values make the scale explicit
        ax.set_ylim(0, 1.02 if survival else min(1.0, max(0.2, top * 1.35)))
        ax.set_xticks(TICKS)
        ax.set_ylabel("Survival probability" if survival else "Cumulative incidence")
        kind = "Kaplan–Meier" if survival else "cumulative incidence (death as competing risk)"
        ax.set_title(f"{ENDPOINT_PAPER_NAME[ep]} — {kind}", fontsize=10, color=INK, pad=6, loc="left")
        ax.legend(fontsize=8, loc="lower right" if survival else "upper left")
        _at_risk(ax, rows, xlabel="Years from treatment")
    fig.suptitle("Outcomes by cohort (counts in the legend: events / patients)",
                 x=0.02, ha="left", fontsize=12, color=INK)
    fig.tight_layout(rect=[0.03, 0.02, 1, 0.95], h_pad=5.5)
    p = out / "fig_km_cohort.png"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"   {p}")


def _arms(run: Path, imaging: str) -> dict:
    out = {}
    for d in sorted(p for p in run.iterdir() if (p / "predictions").is_dir()):
        out[f"combined_A:{imaging}" if d.name == "combined_A" else d.name] = d / "predictions"
    return out


LABEL = {"clinical_only": "Clinical only", "ctfm_st": "CT-FM-ST", "ctfm_lp": "CT-FM-LP",
         "segres_e2e": "SegRes-E2E", "segres_ft": "SegRes-FT", "spectre": "SPECTRE",
         "combined_A:ctfm_st": "Clinical + CT-FM", "combined_A:segres_ft": "Clinical + SegRes-FT",
         "combined_A:segres_e2e": "Clinical + SegRes-E2E"}


def fig_tertiles(runs, spec: str, out: Path):
    """Held-out test set by predicted-risk third, per cohort x endpoint x model."""
    sources = {}
    for r in runs:
        info = json.loads((r / "run_info.json").read_text()) if (r / "run_info.json").exists() else {}
        for arm, pred in _arms(r, info.get("imaging", "ctfm_st")).items():
            sources.setdefault(arm, pred)
    arms = [x for x in LABEL if x in sources] + sorted(set(sources) - set(LABEL))

    def single_fit(arm):                 # 5-seed models have no single risk score to cut
        f = next(iter(sorted(sources[arm].glob("*_fullshot.npz"))), None)
        if f is None:
            return True
        z = np.load(f, allow_pickle=True)
        return "n_seeds" not in z.files or int(z["n_seeds"]) == 1
    multi = [x for x in arms if not single_fit(x)]
    if multi:
        print(f"   tertile KM skipped for {', '.join(LABEL.get(x, x) for x in multi)}: "
              f"fitted with 5 seeds, so there is no single risk score to split into thirds")
    arms = [x for x in arms if x not in multi]
    if not arms:
        return
    for co in COHORTS:
        fig, axes = plt.subplots(len(ENDPOINTS), len(arms),
                                 figsize=(3.1 * len(arms), 3.5 * len(ENDPOINTS)),
                                 squeeze=False, sharex=True, sharey=True)
        for i, ep in enumerate(ENDPOINTS):
            for j, arm in enumerate(arms):
                ax = axes[i][j]
                f = sources[arm] / f"{co}_{ep}_{spec}_fullshot.npz"
                if not f.exists():
                    ax.text(0.5, 0.5, "not run", ha="center", va="center",
                            transform=ax.transAxes, color=MUTED)
                    continue
                z = np.load(f, allow_pickle=True)
                t, e = z["times"].astype(float) / DAYS, z["events"].astype(int)
                grp = np.digitize(z["risk"].astype(float),
                                  np.quantile(z["risk"].astype(float), [1 / 3, 2 / 3]))
                rows = []
                for g in (2, 1, 0):                   # draw high risk last = on top
                    m = grp == g
                    if m.sum() < 3:
                        continue
                    km = KaplanMeierFitter().fit(t[m], e[m])
                    x = km.survival_function_.index.values
                    y = km.survival_function_.iloc[:, 0].values
                    keep = x <= HORIZON
                    ax.step(np.r_[x[keep], HORIZON], np.r_[y[keep], y[keep][-1]], where="post",
                            color=TERTILE_COLOR[g], lw=2, zorder=3 + g,
                            label=f"{TERTILE_NAME[g]} ({int(m.sum())})")
                    rows.append((TERTILE_NAME[g], TERTILE_COLOR[g], t[m]))
                try:
                    p = float(multivariate_logrank_test(t, grp, e).p_value)
                    ax.annotate(f"log-rank p = {p:.2g}", (0.04, 0.06), xycoords="axes fraction",
                                fontsize=7.5, color=INK)
                except Exception:
                    pass
                ax.set_xlim(0, HORIZON); ax.set_ylim(0, 1.02); ax.set_xticks(TICKS)
                if i == 0:
                    ax.set_title(LABEL.get(arm, arm), fontsize=10, color=INK, pad=8)
                if j == 0:
                    ax.set_ylabel(f"{ENDPOINT_PAPER_NAME[ep]}\nsurvival probability", fontsize=9)
                ax.legend(fontsize=7, loc="lower left", bbox_to_anchor=(0, 0.12))
                _at_risk(ax, rows[::-1],        # drawn high->low, listed low->high
                         xlabel="Years from treatment" if i == len(ENDPOINTS) - 1 else None)
        fig.suptitle(f"{co} held-out test set by predicted-risk third (full shot)",
                     x=0.02, ha="left", fontsize=12, color=INK)
        fig.tight_layout(rect=[0.02, 0.02, 1, 0.96], h_pad=5.0)
        p = out / f"fig_km_tertiles_{co.lower()}.png"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"   {p}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", nargs="*", default=[], help="run folders; omit for the cohort figure only")
    ap.add_argument("--spec", choices=["primary", "ajcc"], default="primary")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    runs = [Path(r).resolve() for r in a.runs]
    out = Path(a.out) if a.out else (runs[0] / "_report" / "figures" if runs else Path("figures_km"))
    out.mkdir(parents=True, exist_ok=True)
    fig_cohort(out)
    if runs:
        fig_tertiles(runs, a.spec, out)


if __name__ == "__main__":
    main()
