import pytest
from tacb import make_instance


@pytest.mark.parametrize("K,m", [(2, 1), (2, 2), (5, 1), (5, 3), (8, 8)])
def test_frontier_size_is_controlled(K, m):
    for seed in range(20):
        inst = make_instance(K, m, 0.5, 0.4, rng=seed)
        assert inst.K == K and inst.m == m
