"""Regret curves and exponent fitting (log-log regression)."""
from dataclasses import dataclass
import numpy as np


@dataclass
class RegretCurve:
    horizons: np.ndarray
    mean: np.ndarray
    sem: np.ndarray

    def exponent(self, **kw):
        return fit_exponent(self.horizons, self.mean, **kw)


def fit_exponent(horizons, regret, min_regret=1e-9):
    """Fit regret ~ C * T^e by least squares on (log T, log regret). Returns (e, C, r2, slope_se)."""
    h = np.asarray(horizons, float)
    r = np.asarray(regret, float)
    ok = r > min_regret
    if ok.sum() < 3:
        return float("nan"), float("nan"), float("nan"), float("nan")
    x, y = np.log(h[ok]), np.log(r[ok])
    A = np.vstack([x, np.ones_like(x)]).T
    (e, c), res, *_ = np.linalg.lstsq(A, y, rcond=None)
    fit = A @ np.array([e, c])
    ss_res, ss_tot = ((y - fit) ** 2).sum(), ((y - y.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0
    se = np.sqrt(ss_res / max(len(x) - 2, 1) / ((x - x.mean()) ** 2).sum())
    return float(e), float(np.exp(c)), float(r2), float(se)
