import numpy as np
from .base import Policy, StatsMixin


class UCB1(Policy, StatsMixin):
    """Pull each arm once, then argmax of empirical mean + c * sigma * sqrt(2 log(t+2) / n_i)."""
    anytime = True

    def __init__(self, c: float = 1.0):
        self.c = c
        self.name = f"UCB1(c={c:g})"

    def reset(self, ctx):
        self._init_stats(ctx)

    def select(self, t):
        K = self.ctx.K
        if t <= K:
            return np.full(self.ctx.n_runs, t - 1, dtype=np.int64)
        index = self.sums / self.counts + self.c * self.ctx.sigma * np.sqrt(2.0 * np.log(t + 2.0) / self.counts)
        return np.argmax(index, axis=1)

    def update(self, t, arms, rewards):
        self._record(arms, rewards)
