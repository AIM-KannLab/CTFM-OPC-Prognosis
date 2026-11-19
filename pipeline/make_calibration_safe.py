"""make_calibration.py with one change: a metric that cannot be estimated on a test set (e.g. the
calibration slope does not converge with 4 events in Maastro LR/DM) is reported as NaN instead of
stopping the whole table. IBS, calibration error and everything else are make_calibration.py unchanged.

    python3 runner.py run --data <d> --root <root> -- make_calibration_safe.py --runs <A1> <BASE> --out <dir>
"""
import numpy as np

import make_calibration as MC

_orig = MC.bootstrap_ci


def _safe(fn, t, e, **kw):
    try:
        return _orig(fn, t, e, **kw)
    except (ValueError, ZeroDivisionError) as ex:
        print(f"   !! not estimable on this test set ({ex}) -> NaN")
        return np.nan, np.nan, np.nan, 0


MC.bootstrap_ci = _safe          # workers are forked after this, so they see it too

if __name__ == "__main__":
    MC.main()
