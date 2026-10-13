"""The model: an immutable problem instance. No simulation state lives here."""
from dataclasses import dataclass
from functools import cached_property
import numpy as np
from .frontier import find_frontier

ATTENTION, TIME, CONTESTED = "attention-dominant", "time-dominant", "contested"


def sigmoid(x):
    """Numerically stable logistic function, vectorised."""
    x = np.asarray(x, float)
    e = np.exp(-np.abs(x))
    return np.where(x >= 0, 1.0 / (1.0 + e), e / (1.0 + e))


@dataclass(eq=False)
class Instance:
    """K independent arms, shared exponents: mu_i(t, n) = L_i + D_i * S(n**alpha - t**beta)."""
    L: np.ndarray
    U: np.ndarray
    alpha: float
    beta: float
    sigma: float = 1.0          # Gaussian noise std (known to learners, as is standard)

    def __post_init__(self):
        self.L = np.asarray(self.L, float)
        self.U = np.asarray(self.U, float)
        if self.L.shape != self.U.shape or self.L.ndim != 1 or len(self.L) < 2:
            raise ValueError("L and U must be 1-D arrays of equal length K >= 2")
        if not np.all((self.L > 0) & (self.L < self.U)):
            raise ValueError("need 0 < L_i < U_i for every arm")
        if not (0.0 < self.alpha < 1.0 and 0.0 < self.beta < 1.0):
            raise ValueError("this package covers alpha, beta in the open interval (0, 1)")
        if self.sigma <= 0:
            raise ValueError("sigma must be positive")

    # ---- structure
    @property
    def K(self):
        return len(self.L)

    @property
    def D(self):
        return self.U - self.L

    @property
    def region(self):
        if self.alpha > self.beta:
            return ATTENTION
        if self.alpha < self.beta:
            return TIME
        return CONTESTED

    @cached_property
    def frontier(self):
        return find_frontier(self.L, self.U)

    @property
    def m(self):
        return len(self.frontier)

    @property
    def kind(self):
        return "dominance" if self.m == 1 else "crossover"

    # ---- mean reward, vectorised over any broadcastable (arms, n)
    def mean(self, arms, n, t):
        arms = np.asarray(arms)
        n = np.asarray(n, float)
        return self.L[arms] + self.D[arms] * sigmoid(n ** self.alpha - float(t) ** self.beta)

    def describe(self):
        return (f"K={self.K} m={self.m} ({self.kind}) alpha={self.alpha:.3g} beta={self.beta:.3g} "
                f"[{self.region}]")
