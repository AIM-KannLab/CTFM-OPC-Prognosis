"""The 2yr / 3yr derived endpoint columns, following the original data files' own rule.

Verified against the untouched new_radcure_data.csv and new_maastro_data.csv, every
2yr_/3yr_ column follows:

    event cell = "censored"  if follow-up ended before the horizon without an event
                 "1"         if the event happened by the horizon
                 "0"         otherwise
    time cell  = "censored"  in exactly the rows whose event cell is "censored",
                 the base time (unchanged, not capped) otherwise

All three corrected cohort files build their LR / DM derived columns with this one
function, so they follow the same convention. These columns are not read by any of the
revision analyses -- they are kept for completeness and for older scripts.
"""
HORIZONS = {"2yr": 730, "3yr": 1095}
CENSORED = "censored"


def horizon_cells(time_text: str, event: int, horizon: float):
    """Return (event_cell, time_cell) as text for one patient and one horizon."""
    t = float(time_text)
    if t < horizon and event == 0:
        return CENSORED, CENSORED
    return ("1" if (event == 1 and t <= horizon) else "0"), time_text
