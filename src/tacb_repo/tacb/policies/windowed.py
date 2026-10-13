"""Windowed UCB variants (general K): SW-UCB, RAW-UCB, SSW-UCB. All use per-arm prefix sums.
Bonuses are scaled by the known noise level sigma (the sub-Gaussian scale)."""
import numpy as np
from .base import Policy, PrefixMixin


class SWUCB(Policy, PrefixMixin):
    """Sliding-window UCB: index from the last `tau` pulls of each arm (Garivier & Moulines 2011)."""
    anytime = True

    def __init__(self, tau: int = 50, xi: float = 1.0):
        self.tau, self.xi = int(tau), xi
        self.name = f"SW-UCB(tau={self.tau})"

    def reset(self, ctx):
        self._init_prefix(ctx)

    def select(self, t):
        if t <= self.ctx.K:
            return self._init_round(t)
        n = self.counts.astype(np.int64)
        w = np.minimum(n, self.tau)
        mean = (self._P(n) - self._P(n - w)) / w
        bonus = np.sqrt(self.xi * np.log(min(t, self.tau) + 2.0) / w)
        return np.argmax(mean + self.ctx.sigma * bonus, axis=1)

    def update(self, t, arms, rewards):
        self._record(arms, rewards)


class RAWUCB(Policy, PrefixMixin):
    """Efficient RAW-UCB (Seznec et al. 2020): the most pessimistic index over a geometric grid of
    window lengths 1, 2, 4, ... plus the full window."""
    anytime = True

    def __init__(self, alpha: float = 1.0):
        self.a = alpha
        self.name = "RAW-UCB"

    def reset(self, ctx):
        self._init_prefix(ctx)

    def select(self, t):
        if t <= self.ctx.K:
            return self._init_round(t)
        n = self.counts.astype(np.int64)
        s2 = 2.0 * self.ctx.sigma ** 2 * self.a * np.log(t + 2.0)
        Pn = self._P(n)
        best = Pn / n + np.sqrt(s2 / n)                    # full window
        w = 1
        while w <= n.max():
            ok = w <= n
            idx = (Pn - self._P(np.maximum(n - w, 0))) / w + np.sqrt(s2 / w)
            best = np.where(ok, np.minimum(best, idx), best)
            w *= 2
        return np.argmax(best, axis=1)

    def update(self, t, arms, rewards):
        self._record(arms, rewards)


class SSWUCB(Policy, PrefixMixin):
    """Saturation-gated sliding window: use the full-history mean unless the recent window differs
    significantly from it (z-test), in which case use the recent window."""
    anytime = True

    def __init__(self, short_frac: float = 0.25, z_thresh: float = 2.0, alpha: float = 1.0):
        self.f, self.z, self.a = short_frac, z_thresh, alpha
        self.name = "SSW-UCB"

    def reset(self, ctx):
        self._init_prefix(ctx)

    def select(self, t):
        if t <= self.ctx.K:
            return self._init_round(t)
        n = self.counts.astype(np.int64)
        sg = self.ctx.sigma
        Pn = self._P(n)
        full = Pn / n
        sw = np.maximum(1, (self.f * n).astype(np.int64))
        short = (Pn - self._P(n - sw)) / sw
        z = np.abs(short - full) / (sg * np.sqrt(1.0 / sw + 1.0 / n))
        log_term = self.a * np.log(t + 2.0)
        use_full = z < self.z
        mean = np.where(use_full, full, short)
        w = np.where(use_full, n, sw)
        return np.argmax(mean + np.sqrt(2.0 * sg ** 2 * log_term / w), axis=1)

    def update(self, t, arms, rewards):
        self._record(arms, rewards)
