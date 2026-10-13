import numpy as np
from .base import Policy


class FixedSequence(Policy):
    """Plays a predetermined arm sequence in every run (ignores rewards)."""
    anytime = False

    def __init__(self, sequence, name="FixedSequence"):
        self.sequence = np.asarray(sequence, dtype=np.int64)
        self.name = name

    def reset(self, ctx):
        self.R = ctx.n_runs
        assert len(self.sequence) >= ctx.T

    def select(self, t):
        return np.full(self.R, self.sequence[t - 1], dtype=np.int64)

    def update(self, t, arms, rewards):
        pass


class OraclePolicy(FixedSequence):
    """Knows the instance: replays the oracle's optimal sequence for the horizon. Regret must be 0."""

    def __init__(self, oracle):
        super().__init__([0], name="Oracle")
        self.oracle = oracle

    def reset(self, ctx):
        self.sequence = self.oracle.solve(ctx.T).sequence()
        super().reset(ctx)
