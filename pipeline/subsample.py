"""Few-shot training subsets and shot grid, matching the published pipeline.

Both functions are ports of `working_space/few_shot_single_institution_running.py`
so that for a given (shot, seed) the training subset is identical to the paper's.
"""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

# Verbatim from line 659 of the published script.
FEWSHOT_SIZES_RAW = [10, 24, 50, 100, 150, 200, 0.10, 0.20, 0.40, 0.60, 0.80, 1.00]

# The manuscript states 5 seeds. The CT-FM-ST script uses range(10), the PyTorch-head
# script range(5) -- a code/text discrepancy logged for A7. A1 uses 5 uniformly so the
# SD columns are comparable across arms.
SEEDS = list(range(5))


def shot_grid(n_train: int) -> list[int]:
    """Distinct training-set sizes the published script actually runs, for a cohort.

    Ports lines 670-684: integer shots are clamped to the training-split size;
    fractional shots are SKIPPED when they resolve to <= the largest integer shot
    (200), because those sizes are already covered by the integer entries.

    Radcure (940) -> [10, 24, 50, 100, 150, 200, 376, 564, 752, 940]
    Maastro (121) -> [10, 24, 50, 100, 121]      (150 and 200 both clamp to 121)
    """
    max_int = max(s for s in FEWSHOT_SIZES_RAW if isinstance(s, int))
    sizes = set()
    for shot in FEWSHOT_SIZES_RAW:
        if isinstance(shot, int):
            size = min(n_train, shot)
        else:
            size = max(1, int(n_train * shot))
            if max_int > 0 and size <= max_int:
                continue
        if size > 0:
            sizes.add(int(size))
    return sorted(sizes)


def stratified_subsample(df: pd.DataFrame, n_samples: int, seed: int) -> pd.DataFrame:
    """Stratified subset of size n_samples; random fallback if stratification fails.

    Stratifies on `event` only. When n_samples >= len(df) the full frame is returned
    unchanged -- which is why full-shot results carry no seed-to-seed variance.
    """
    if n_samples >= len(df):
        return df.reset_index(drop=True)
    try:
        splitter = StratifiedShuffleSplit(n_splits=1, train_size=n_samples, random_state=seed)
        subset_idx, _ = next(splitter.split(df, df["event"]))
        return df.iloc[subset_idx].reset_index(drop=True)
    except ValueError:
        return df.sample(n=n_samples, random_state=seed, replace=False).reset_index(drop=True)
