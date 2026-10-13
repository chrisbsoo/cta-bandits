"""Policy interface. A policy sees only rewards (never L, U, alpha, beta) and is vectorised over runs."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np


@dataclass
class Context:
    n_runs: int
    K: int
    T: int                       # horizon (anytime policies must ignore it)
    sigma: float                 # noise level, known to the learner
    rng: np.random.Generator
    alpha: float = None          # exponents are passed ONLY for policies that assume them (GLR-UCB);
    beta: float = None           # every other policy must ignore them


class Policy(ABC):
    name = "policy"
    anytime = True               # False if behaviour depends on the horizon T (e.g. ETC)

    @abstractmethod
    def reset(self, ctx: Context) -> None: ...

    @abstractmethod
    def select(self, t: int) -> np.ndarray:
        """Arms for round t (1-based), integer array of shape (n_runs,)."""

    @abstractmethod
    def update(self, t: int, arms: np.ndarray, rewards: np.ndarray) -> None: ...


class StatsMixin:
    """Per-arm pull counts and reward sums, shape (n_runs, K)."""

    def _init_stats(self, ctx: Context):
        self.ctx = ctx
        self._rows = np.arange(ctx.n_runs)
        self.counts = np.zeros((ctx.n_runs, ctx.K))
        self.sums = np.zeros((ctx.n_runs, ctx.K))

    def _record(self, arms, rewards):
        self.counts[self._rows, arms] += 1
        self.sums[self._rows, arms] += rewards


class PrefixMixin(StatsMixin):
    """Adds per-arm prefix sums of rewards, shape (n_runs, K, T+1), so any window mean is O(1):
    mean of the last w pulls of arm k = (P[n_k] - P[n_k - w]) / w."""

    def _init_prefix(self, ctx: Context):
        self._init_stats(ctx)
        self.prefix = np.zeros((ctx.n_runs, ctx.K, ctx.T + 1))
        self._cols = np.arange(ctx.K)[None, :]

    def _record(self, arms, rewards):
        n_new = self.counts[self._rows, arms].astype(np.int64) + 1
        self.prefix[self._rows, arms, n_new] = self.prefix[self._rows, arms, n_new - 1] + rewards
        self.counts[self._rows, arms] += 1
        self.sums[self._rows, arms] += rewards

    def _P(self, idx):
        """Gather prefix sums at per-(run, arm) positions idx, shape (n_runs, K)."""
        return self.prefix[self._rows[:, None], self._cols, idx]

    def _init_round(self, t):
        """Round-robin first K rounds so that every arm has been pulled once."""
        return np.full(self.ctx.n_runs, t - 1, dtype=np.int64)
