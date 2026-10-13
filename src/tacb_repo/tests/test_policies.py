import math
import numpy as np
import pytest
from tacb import POLICIES, make_policy, make_instance, Simulator, Context, GLRUCB
from tacb.model import sigmoid
import reference as ref

HORIZON = 60


def drive(policy, run_ids, T=HORIZON, K=4, alpha=0.6, beta=0.4):
    """Run `policy` with deterministic rewards r(run, arm, count); return chosen arms, shape (len(run_ids), T)."""
    R = len(run_ids)
    policy.reset(Context(n_runs=R, K=K, T=T, sigma=1.0, rng=np.random.default_rng(0), alpha=alpha, beta=beta))
    counts = np.zeros((R, K), dtype=int); out = np.zeros((R, T), dtype=int)
    for t in range(1, T + 1):
        arms = policy.select(t)
        n = counts[np.arange(R), arms]
        rewards = 5.0 + 2.0 * np.sin(1.0 + 7.0 * np.array(run_ids) + 3.0 * arms + 0.37 * n) + arms
        policy.update(t, arms, rewards); counts[np.arange(R), arms] += 1; out[:, t - 1] = arms
    return out


@pytest.mark.parametrize("name", sorted(POLICIES))
def test_runs_are_independent(name):
    """Batched decisions for run r equal the decisions of the same policy run alone (no cross-run leakage)."""
    batch = drive(make_policy(name), [0, 1, 2, 3])
    for r in range(4):
        assert np.array_equal(batch[r], drive(make_policy(name), [r])[0]), f"{name}: run {r} differs"


@pytest.mark.parametrize("name", ["ucb1", "greedy", "swucb", "rawucb", "sswucb"])
def test_first_K_rounds_pull_each_arm_once(name):
    arms = drive(make_policy(name), [0], K=4)[0]
    assert list(arms[:4]) == [0, 1, 2, 3]


def _history(rng, n, T):
    p = np.zeros(T + 1); p[1:n + 1] = np.cumsum(rng.normal(5, 2, n)); return p


@pytest.mark.parametrize("name,scalar", [("ucb1", None), ("swucb", ref.sw_index), ("rawucb", ref.raw_index),
                                         ("sswucb", ref.ssw_index)])
def test_port_matches_original_formulas_for_K2(name, scalar):
    """Vectorised decision == scalar re-statement of the original numba rule, on random histories."""
    rng = np.random.default_rng(1)
    for _ in range(200):
        T = 400; n = rng.integers(1, 120, 2); t = int(n.sum()) + 1
        P = [_history(rng, int(k), T) for k in n]
        pol = make_policy(name) if name != "swucb" else make_policy(name, tau=50)
        pol.reset(Context(n_runs=1, K=2, T=T, sigma=1.0, rng=np.random.default_rng(0)))
        pol.counts[0] = n
        pol.sums[0] = [P[0][n[0]], P[1][n[1]]]
        if hasattr(pol, "prefix"):
            pol.prefix[0, 0], pol.prefix[0, 1] = P[0], P[1]
        got = int(pol.select(t)[0])
        idx = [ref.ucb1_index(int(n[k]), P[k][n[k]], t) if scalar is None else scalar(int(n[k]), P[k], t) for k in range(2)]
        want = int(np.argmax(idx))
        assert got == want


def test_glr_decision_matches_original_K2():
    rng = np.random.default_rng(2); a, b, T = 0.7, 0.3, 500
    for _ in range(100):
        pol = GLRUCB()
        pol.reset(Context(n_runs=1, K=2, T=T, sigma=1.0, rng=np.random.default_rng(0), alpha=a, beta=b))
        ns = rng.integers(3, 60, 2); XtX = np.zeros((2, 2, 2)); Xty = np.zeros((2, 2)); sums = np.zeros(2)
        for k in range(2):
            for j in range(int(ns[k])):
                w = float(sigmoid(j ** a - (j + 1 + 5 * k + 20) ** b)); x = np.array([1 - w, w]); r = rng.normal(8, 3)
                XtX[k] += np.outer(x, x); Xty[k] += x * r; sums[k] += r
        pol.XtX[0], pol.Xty[0], pol.sums[0], pol.n[0] = XtX, Xty, sums, ns
        t = int(ns.sum()) + 1
        arms = pol.select(t)
        arm, committed = ref.glr_decide(sums, ns, XtX, Xty, pol.A, pol.B, 1.0, t)
        assert int(arms[0]) == arm and (pol.committed[0] >= 0) == committed


def test_glr_requires_exponents():
    with pytest.raises(ValueError):
        GLRUCB().reset(Context(n_runs=1, K=2, T=10, sigma=1.0, rng=np.random.default_rng(0)))


def test_glr_commit_rule_picks_the_right_arm_under_balanced_sampling():
    """With both arms sampled equally the GLR test fires and picks the arm with the larger F.
    (Under GLR's own UCB1 fallback the losing arm can be starved and the test may never fire;
    that is a property of the algorithm, documented in the README, not of this port.)"""
    from tacb import Environment, Instance
    base = make_instance(2, 2, 0.7, 0.3, rng=1); T, R = 1000, 30
    inst = Instance(base.L, base.U, 0.7, 0.3, 1.0)
    env = Environment(inst, R, np.random.default_rng(0)); pol = GLRUCB()
    pol.reset(Context(n_runs=R, K=2, T=T, sigma=1.0, rng=np.random.default_rng(0), alpha=0.7, beta=0.3))
    for t in range(1, T + 1):
        arms = np.full(R, (t - 1) % 2); rewards, _ = env.step(arms, t); pol.update(t, arms, rewards)
        if t % 50 == 0:
            pol.select(t)
    best = int(np.argmax([sum(inst.mean(k, u - 1, u) for u in range(1, T + 1)) for k in (0, 1)]))
    assert (pol.committed >= 0).mean() > 0.9 and (pol.committed[pol.committed >= 0] == best).mean() > 0.9
