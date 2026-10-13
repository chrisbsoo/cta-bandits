"""Algorithm 1 (FindFrontier). Pure function, O(K log K)."""
import numpy as np


def find_frontier(L, U):
    """Indices of the arms that no other arm dominates, ordered by INCREASING floor.

    Sort by L descending (ties: U descending); scan with a running max of U; keep an arm iff its
    U strictly exceeds the running max. The strict test also drops identical copies after the first.
    """
    L = np.asarray(L, float)
    U = np.asarray(U, float)
    order = np.lexsort((-U, -L))
    keep, running = [], -np.inf
    for i in order:
        if U[i] > running:
            keep.append(int(i))
            running = U[i]
    return np.array(keep[::-1], dtype=np.int64)
