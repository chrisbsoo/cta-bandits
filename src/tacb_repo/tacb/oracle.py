"""The oracle benchmark V*(T) = max over pull sequences of the cumulative mean reward.

Oracle.solve() applies the case table (Section 2.5) and uses the cheapest exact method:
    m == 1                    -> constant on the dominating arm           (Prop. 8)
    m >= 2, alpha >= beta     -> constant on the best frontier arm        (Thm. 1)
    m >= 2, alpha <  beta     -> one block per frontier arm (block DP)    (Thm. 2, Cor. 2)
"""
from dataclasses import dataclass, field
import numpy as np
from .model import Instance
from ._kernels import constant_cumsum, block_dp


@dataclass
class OracleResult:
    value: float                 # V*(T)
    method: str                  # 'dominance' | 'no-switching' | 'blocks'
    horizon: int
    blocks: list = field(default_factory=list)      # [(arm, start, length)]

    def sequence(self):
        seq = np.empty(self.horizon, dtype=np.int64)
        for arm, start, length in self.blocks:
            seq[start:start + length] = arm
        return seq


class Oracle:
    def __init__(self, instance: Instance):
        self.inst = instance
        self._cache = {}

    @property
    def method(self):
        if self.inst.m == 1:
            return "dominance"
        return "no-switching" if self.inst.alpha >= self.inst.beta else "blocks"

    def _prefix(self, T):
        """F_i(1..T) for every frontier arm: shape (m, T)."""
        i = self.inst
        return np.stack([constant_cumsum(i.L[k], i.U[k], i.alpha, i.beta, T) for k in i.frontier])

    def solve(self, T: int) -> OracleResult:
        T = int(T)
        if T in self._cache:
            return self._cache[T]
        i = self.inst
        if self.method != "blocks":
            F = self._prefix(T)[:, -1]
            j = int(np.argmax(F))
            res = OracleResult(float(F[j]), self.method, T, [(int(i.frontier[j]), 0, T)])
        else:
            fr = i.frontier
            val, parent = block_dp(i.L[fr], i.U[fr], i.alpha, i.beta, T)
            blocks, t = [], T
            for r in range(len(fr) - 1, -1, -1):
                s = int(parent[r, t])
                if s >= 0:
                    blocks.append((int(fr[r]), s, t - s))
                    t = s
            blocks.reverse()
            res = OracleResult(float(val), "blocks", T, blocks)
        self._cache[T] = res
        return res

    def values(self, horizons):
        """V*(T') for each horizon; one pass when no-switching applies, one DP per horizon otherwise."""
        horizons = np.asarray(horizons, int)
        if self.method != "blocks":
            pre = self._prefix(int(horizons.max())).max(axis=0)
            return pre[horizons - 1]
        return np.array([self.solve(int(h)).value for h in horizons])
