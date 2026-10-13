"""Instance generators with a CONTROLLED frontier size m."""
import numpy as np
from .model import Instance


def make_instance(K, m, alpha, beta, rng=None, sigma=1.0):
    """K arms of which exactly m are on the frontier (floors increasing, ceilings decreasing);
    the other K-m arms are strictly dominated by a frontier arm."""
    if not 1 <= m <= K:
        raise ValueError("need 1 <= m <= K")
    rng = np.random.default_rng(rng)
    Lf = np.sort(rng.uniform(1.0, 6.0, m))
    Uf = np.sort(rng.uniform(8.0, 20.0, m))[::-1]
    L, U = list(Lf), list(Uf)
    for _ in range(K - m):
        f = rng.integers(0, m)
        L.append(Lf[f] - rng.uniform(0.1, 0.8) * Lf[f])
        U.append(Uf[f] - rng.uniform(0.05, 0.5) * (Uf[f] - Lf[f]))
    perm = rng.permutation(K)
    inst = Instance(np.array(L)[perm], np.array(U)[perm], alpha, beta, sigma)
    assert inst.m == m, "generator produced the wrong frontier size"
    return inst

def make_nested(K, m, alpha, beta, rng=None, sigma=1.0, gap_L=0.5, gap_U=1.0, L0=1.0, U0=20.0):
    """Nested fixed-gap instances: frontier arm i has floor L0 + i*gap_L and ceiling U0 - i*gap_U, so the
    m-instance is the (m-1)-instance plus ONE more arm and neighbouring gaps do not depend on m.
    The K-m remaining arms are strictly dominated by arm 0 (always on the frontier). `rng` only shuffles arm order."""
    if not 1 <= m <= K:
        raise ValueError("need 1 <= m <= K")
    rng = np.random.default_rng(rng)
    L = [L0 + i * gap_L for i in range(m)]
    U = [U0 - i * gap_U for i in range(m)]
    for j in range(K - m):
        L.append(L0 - 0.3)
        U.append(U0 - 0.5 * (j + 1))
    perm = rng.permutation(K)
    inst = Instance(np.array(L)[perm], np.array(U)[perm], alpha, beta, sigma)
    assert inst.m == m, "nested generator produced the wrong frontier size"
    return inst