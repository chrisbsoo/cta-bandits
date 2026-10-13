import numpy as np
from .base import Policy
from ..model import sigmoid


class GLRUCB(Policy):
    """GLR-UCB for the no-switching regime. ASSUMES THE EXPONENTS (alpha, beta) ARE KNOWN.

    Each arm's rewards are regressed on x_t = (1 - w_t, w_t), w_t = S(n^alpha - t^beta); the coefficients
    estimate (L_i, U_i), and F_i = A*L_i + B*U_i is the value of pulling arm i for the whole horizon, with
    A = sum_t (1 - w_t), B = sum_t w_t along the sustained-play path. A GLR confidence boundary on
    F_best - F_j for every rival j (union bound over K-1 rivals) triggers COMMITMENT to the best arm;
    until then UCB1 decides. For K = 2 this is the original test. Commitment is only valid where constant
    play is optimal (alpha >= beta); the horizon enters through A, B, so the policy is not anytime.
    """
    anytime = False

    def __init__(self, delta=0.05, c1=0.8, c2=1.6, min_floor=3, ucb_c=1.0, ridge=1e-6):
        self.delta, self.c1, self.c2 = delta, c1, c2
        self.min_floor, self.ucb_c, self.ridge = int(min_floor), ucb_c, ridge
        self.name = "GLR-UCB"

    def reset(self, ctx):
        if ctx.alpha is None or ctx.beta is None:
            raise ValueError("GLR-UCB needs the exponents in the Context")
        self.ctx = ctx
        R, K = ctx.n_runs, ctx.K
        t = np.arange(1, ctx.T + 1, dtype=float)
        w = sigmoid((t - 1.0) ** ctx.alpha - t ** ctx.beta)
        self.A, self.B = float((1 - w).sum()), float(w.sum())
        self._rows = np.arange(R)
        self.n = np.zeros((R, K))
        self.sums = np.zeros((R, K))
        self.XtX = np.zeros((R, K, 2, 2))
        self.Xty = np.zeros((R, K, 2))
        self.committed = np.full(R, -1, dtype=np.int64)
        self.delta_k = self.delta / max(K - 1, 1)

    def select(self, t):
        R, K = self.ctx.n_runs, self.ctx.K
        arms = np.empty(R, dtype=np.int64)
        done = self.committed >= 0
        arms[done] = self.committed[done]
        live = ~done
        if live.any():
            low = self.n < self.min_floor
            need = live & low.any(axis=1)
            arms[need] = np.argmax(low[need], axis=1)          # first under-sampled arm
            test = np.where(live & ~need)[0]
            if len(test):
                self._test(test, t, arms)
        return arms

    def _test(self, idx, t, arms):
        sg = self.ctx.sigma
        M = self.XtX[idx] + self.ridge * np.eye(2)
        det = M[..., 0, 0] * M[..., 1, 1] - M[..., 0, 1] * M[..., 1, 0]
        inv = np.empty_like(M)
        inv[..., 0, 0] = M[..., 1, 1] / det; inv[..., 0, 1] = -M[..., 0, 1] / det
        inv[..., 1, 0] = -M[..., 1, 0] / det; inv[..., 1, 1] = M[..., 0, 0] / det
        beta = np.einsum("rkij,rkj->rki", inv, self.Xty[idx])
        F = self.A * beta[..., 0] + self.B * beta[..., 1]
        c = np.array([self.A, self.B])
        V = sg ** 2 * np.einsum("i,rkij,j->rk", c, inv, c)
        r = np.arange(len(idx))
        best = np.argmax(F, axis=1)
        diff = F[r, best][:, None] - F
        Vs = V[r, best][:, None] + V
        Vb = np.where(Vs > 1, Vs, 1.0001)
        bound = np.sqrt(2 * Vb * (self.c1 * np.log(np.log(Vb) + np.e) + self.c2 * np.log(1.0 / self.delta_k)))
        sep = diff > bound
        sep[r, best] = True
        commit = sep.all(axis=1)
        self.committed[idx[commit]] = best[commit]
        arms[idx[commit]] = best[commit]
        rest = ~commit
        if rest.any():
            n = self.n[idx[rest]]
            index = self.sums[idx[rest]] / n + self.ucb_c * sg * np.sqrt(2.0 * np.log(t + 2.0) / n)
            arms[idx[rest]] = np.argmax(index, axis=1)

    def update(self, t, arms, rewards):
        n_pull = self.n[self._rows, arms]
        w = sigmoid(n_pull ** self.ctx.alpha - float(t) ** self.ctx.beta)
        x = np.stack([1 - w, w], axis=1)                        # (R, 2)
        self.XtX[self._rows, arms] += x[:, :, None] * x[:, None, :]
        self.Xty[self._rows, arms] += x * rewards[:, None]
        self.sums[self._rows, arms] += rewards
        self.n[self._rows, arms] += 1
