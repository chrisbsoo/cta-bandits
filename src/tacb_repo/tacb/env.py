"""Environment: owns the TRUE state (pull counts) and the noise. Vectorised over independent runs."""
import numpy as np
from .model import Instance


class Environment:
    def __init__(self, instance: Instance, n_runs: int, rng: np.random.Generator):
        self.inst = instance
        self.R = int(n_runs)
        self.rng = rng
        self._rows = np.arange(self.R)
        self.reset()

    def reset(self):
        self.n = np.zeros((self.R, self.inst.K), dtype=np.int64)

    def step(self, arms, t):
        """Pull `arms` (shape (R,)) at round t (1-based). Returns (noisy rewards, true means)."""
        n_pulled = self.n[self._rows, arms]
        mu = self.inst.mean(arms, n_pulled, t)
        rewards = mu + self.inst.sigma * self.rng.standard_normal(self.R)
        self.n[self._rows, arms] += 1
        return rewards, mu
