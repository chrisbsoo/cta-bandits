"""Independent reference implementations used only by the tests."""
from functools import lru_cache
import math


def sig(x):
    return 1 / (1 + math.exp(-x)) if x >= 0 else math.exp(x) / (1 + math.exp(x))


def f(L, U, n, t, a, b):
    return L + (U - L) * sig((n ** a) - t ** b)


def bruteforce(L, U, a, b, T):
    """V*(T) by DP over count vectors, Theta(T^K/(K-1)!)."""
    K = len(L)

    @lru_cache(maxsize=None)
    def W(counts, t):
        if t > T:
            return 0.0
        best = -1e300
        for i in range(K):
            c = list(counts); c[i] += 1
            best = max(best, f(L[i], U[i], counts[i], t, a, b) + W(tuple(c), t + 1))
        return best

    return W(tuple([0] * K), 1)


def sequence_value(L, U, a, b, seq):
    cnt = [0] * len(L); v = 0.0
    for t, i in enumerate(seq, 1):
        v += f(L[i], U[i], cnt[i], t, a, b); cnt[i] += 1
    return v


# ---- scalar re-statements of the ORIGINAL numba decision rules (K = 2), used to check the vectorised ports
import numpy as np


def ucb1_index(n, s, t, c=1.0, sigma=1.0):
    return s / n + c * sigma * math.sqrt(2.0 * math.log(t + 2.0) / n)


def sw_index(n, prefix, t, tau=50, xi=1.0, sigma=1.0):
    w = min(n, tau)
    return (prefix[n] - prefix[n - w]) / w + sigma * math.sqrt(xi * math.log(min(t, tau) + 2.0) / w)


def raw_index(n, prefix, t, alpha=1.0, sigma=1.0):
    log_term = alpha * math.log(t + 2.0)
    best, w = 1e18, 1
    while w <= n:
        best = min(best, (prefix[n] - prefix[n - w]) / w + math.sqrt(2.0 * sigma ** 2 * log_term / w))
        w *= 2
    if w // 2 != n:
        best = min(best, prefix[n] / n + math.sqrt(2.0 * sigma ** 2 * log_term / n))
    return best


def ssw_index(n, prefix, t, sigma=1.0, short_frac=0.25, z_thresh=2.0, alpha=1.0):
    full = prefix[n] / n
    sw = max(1, int(short_frac * n))
    short = (prefix[n] - prefix[n - sw]) / sw
    se = sigma * math.sqrt(1.0 / sw + 1.0 / n)
    z = abs(short - full) / se
    lt = alpha * math.log(t + 2.0)
    return full + math.sqrt(2.0 * sigma ** 2 * lt / n) if z < z_thresh else short + math.sqrt(2.0 * sigma ** 2 * lt / sw)


def glr_decide(sums, ns, XtX, Xty, A, B, sigma, t, delta=0.05, c1=0.8, c2=1.6, ucb_c=1.0, ridge=1e-6):
    """Original K=2 GLR-UCB decision: returns (arm, committed_flag)."""
    c = np.array([A, B]); F, V = [], []
    for k in range(2):
        Mi = np.linalg.inv(XtX[k] + ridge * np.eye(2))
        F.append(c @ Mi @ Xty[k]); V.append(c @ Mi @ c)
    Vv = sigma ** 2 * (V[0] + V[1]); diff = F[0] - F[1]
    Vb = Vv if Vv > 1 else 1.0001
    bound = math.sqrt(2 * Vb * (c1 * math.log(math.log(Vb) + math.e) + c2 * math.log(1.0 / delta)))
    if abs(diff) > bound:
        return (0 if diff > 0 else 1), True
    i0 = sums[0] / ns[0] + ucb_c * sigma * math.sqrt(2.0 * math.log(t + 2.0) / ns[0])
    i1 = sums[1] / ns[1] + ucb_c * sigma * math.sqrt(2.0 * math.log(t + 2.0) / ns[1])
    return (0 if i0 >= i1 else 1), False
