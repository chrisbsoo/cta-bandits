import numpy as np, pytest
from tacb import Instance, Oracle, find_frontier, make_instance
from reference import bruteforce, sequence_value


def random_instance(rng):
    K = int(rng.choice([2, 3, 4])); T = int(rng.integers(5, 12 if K < 4 else 9))
    a, b = rng.uniform(0.02, 0.98, 2)
    u = rng.random()
    if u < 0.2: b = a
    L = rng.uniform(0.1, 10, K); U = L + rng.uniform(0.05, 40, K)
    kind = rng.integers(0, 4)
    if kind == 1 and K >= 3:                                              # duplicate arm + equal floor
        L[1], U[1], L[2] = L[0], U[0], L[0]; U[2] = L[0] + rng.uniform(0.05, 40)
    if kind == 2: L = np.sort(L); U = L.max() + 5 + np.sort(rng.uniform(0, 40, K))[::-1]   # all on frontier
    if kind == 3: U = L + 1 + 3 * L                                       # heavy dominance
    return Instance(L, U, float(a), float(b)), T


def test_oracle_matches_bruteforce_and_returns_optimal_sequence():
    rng = np.random.default_rng(0); methods = set()
    for _ in range(300):
        inst, T = random_instance(rng)
        res = Oracle(inst).solve(T)
        ref = bruteforce(list(inst.L), list(inst.U), inst.alpha, inst.beta, T)
        own = sequence_value(inst.L, inst.U, inst.alpha, inst.beta, res.sequence())
        assert abs(res.value - ref) < 1e-9 and abs(own - ref) < 1e-9
        methods.add(res.method)
    assert methods == {"dominance", "no-switching", "blocks"}


def test_values_over_horizons_matches_single_solves():
    inst = make_instance(4, 3, 0.7, 0.3, rng=1); o = Oracle(inst)
    hs = [3, 10, 25, 60]
    assert np.allclose(o.values(hs), [Oracle(inst).solve(h).value for h in hs])


def test_frontier_basic():
    L = np.array([5, 5, 5, 3, 3, 1]); U = np.array([9, 9, 7, 12, 8, 20])
    # arm0 and arm1 identical; arm2 dominated by arm0; arm4 dominated by arm3; arm5 on frontier
    assert list(find_frontier(L, U)) == [5, 3, 0]          # increasing floor


def test_equal_floors_only_top_ceiling_survives():
    assert list(find_frontier([2, 2, 2], [5, 9, 7])) == [1]
