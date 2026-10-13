import numpy as np
from .base import Policy, StatsMixin


class ETC(Policy, StatsMixin):
    """Explore-then-commit. Explore each arm m = ceil(T**gamma) times ('block': arm after arm;
    'alternate': round-robin), then commit to the arm with the best empirical mean."""
    anytime = False

    def __init__(self, gamma: float = 2 / 3, schedule: str = "block"):
        assert schedule in ("block", "alternate")
        self.gamma, self.schedule = gamma, schedule
        self.name = f"ETC({schedule},gamma={gamma:.3g})"

    def reset(self, ctx):
        self._init_stats(ctx)
        self.m = max(1, min(int(np.ceil(ctx.T ** self.gamma)), ctx.T // ctx.K))
        self.explore_len = self.m * ctx.K
        self.commit = None

    def select(self, t):
        R, K = self.ctx.n_runs, self.ctx.K
        if t <= self.explore_len:
            arm = (t - 1) // self.m if self.schedule == "block" else (t - 1) % K
            return np.full(R, arm, dtype=np.int64)
        if self.commit is None:
            self.commit = np.argmax(self.sums / self.counts, axis=1)
        return self.commit

    def update(self, t, arms, rewards):
        if t <= self.explore_len:
            self._record(arms, rewards)
