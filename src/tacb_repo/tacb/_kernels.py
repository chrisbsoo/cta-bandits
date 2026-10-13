"""Numba kernels: PURE FUNCTIONS ONLY (numba does not mix well with classes; classes call these)."""
import numpy as np
from numba import njit, prange


@njit(cache=True)
def sigmoid_scalar(x):
    if x >= 0.0:
        return 1.0 / (1.0 + np.exp(-x))
    e = np.exp(x)
    return e / (1.0 + e)


@njit(cache=True)
def constant_cumsum(L, U, a, b, T):
    """Prefix values F(1..T) of pulling one arm every round: F(t) = sum_{s<=t} f(s-1, s)."""
    out = np.empty(T)
    v = 0.0
    for t in range(1, T + 1):
        na = 0.0 if t == 1 else float(t - 1) ** a
        v += L + (U - L) * sigmoid_scalar(na - float(t) ** b)
        out[t - 1] = v
    return out


@njit(cache=True, parallel=True)
def block_dp(Ls, Us, a, b, T):
    """Corollary 2: W_r(t) = best value of the first t rounds using the first r arms (floor-increasing),
    each arm in one contiguous block. O(m T^2); start points are split over threads."""
    m = Ls.shape[0]
    NEG = -1.0e300
    NA = np.empty(T + 1)
    TB = np.empty(T + 1)
    for n in range(T + 1):
        NA[n] = 0.0 if n == 0 else float(n) ** a
        TB[n] = float(n) ** b
    W = np.full(T + 1, NEG)
    W[0] = 0.0
    parent = np.full((m, T + 1), -1, dtype=np.int64)
    nch = min(T + 1, 64)
    for r in range(m):
        L = Ls[r]
        D = Us[r] - Ls[r]
        loc_v = np.full((nch, T + 1), NEG)
        loc_p = np.full((nch, T + 1), -1, dtype=np.int64)
        for ch in prange(nch):
            for s in range(ch, T + 1, nch):
                if W[s] <= NEG / 2:
                    continue
                B = 0.0
                Ws = W[s]
                for l in range(1, T - s + 1):
                    B += L + D * sigmoid_scalar(NA[l - 1] - TB[s + l])
                    c = Ws + B
                    if c > loc_v[ch, s + l]:
                        loc_v[ch, s + l] = c
                        loc_p[ch, s + l] = s
        Wn = W.copy()
        for t in range(T + 1):
            for ch in range(nch):
                if loc_v[ch, t] > Wn[t]:
                    Wn[t] = loc_v[ch, t]
                    parent[r, t] = loc_p[ch, t]
        W = Wn
    return W[T], parent
