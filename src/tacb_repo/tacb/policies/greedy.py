import numpy as np
from .base import Policy, StatsMixin


class Greedy(Policy, StatsMixin):
    """Pull each arm once, then always the arm with the best empirical mean."""
    anytime = True
    name = "Greedy"

    def reset(self, ctx):
        self._init_stats(ctx)

    def select(self, t):
        if t <= self.ctx.K:
            return np.full(self.ctx.n_runs, t - 1, dtype=np.int64)
        return np.argmax(self.sums / self.counts, axis=1)

    def update(self, t, arms, rewards):
        self._record(arms, rewards)
