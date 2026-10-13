import numpy as np
from tacb import Instance, Simulator, OraclePolicy, FixedSequence, UCB1, ETC, make_instance
from reference import sequence_value


def test_oracle_policy_has_exactly_zero_regret_in_every_region():
    for a, b in [(0.7, 0.3), (0.5, 0.5), (0.3, 0.7)]:
        inst = make_instance(4, 3, a, b, rng=2)
        sim = Simulator(inst, n_runs=20, seed=0)
        res = sim.run(OraclePolicy(sim.oracle), T=40, checkpoints=[40])
        assert np.allclose(res.regret(sim.oracle), 0.0, atol=1e-9)


def test_fixed_sequence_value_matches_reference():
    inst = make_instance(3, 3, 0.4, 0.6, rng=3); seq = [0, 0, 1, 2, 2, 1, 0, 0, 2, 1]
    res = Simulator(inst, n_runs=5, seed=0).run(FixedSequence(seq), T=10)
    assert np.allclose(res.cum_mean_reward[:, -1], sequence_value(inst.L, inst.U, inst.alpha, inst.beta, seq))


def test_counts_sum_to_T_and_reproducible():
    inst = make_instance(3, 2, 0.6, 0.4, rng=4)
    r1 = Simulator(inst, 30, seed=7).run(UCB1(), 200); r2 = Simulator(inst, 30, seed=7).run(UCB1(), 200)
    assert np.all(r1.final_counts.sum(axis=1) == 200)
    assert np.array_equal(r1.cum_mean_reward, r2.cum_mean_reward)


def test_noise_is_gaussian_with_known_mean():
    from tacb import Environment
    inst = Instance([2.0, 3.0], [5.0, 9.0], 0.5, 0.4, sigma=0.5)
    env = Environment(inst, 20000, np.random.default_rng(0))
    rewards, mu = env.step(np.zeros(20000, dtype=np.int64), 1)
    assert abs((rewards - mu).mean()) < 0.02 and abs((rewards - mu).std() - 0.5) < 0.02


def test_learners_have_positive_regret_and_never_beat_oracle():
    inst = make_instance(3, 3, 0.7, 0.3, rng=5); sim = Simulator(inst, 50, seed=1)
    for pol in (UCB1(), ETC()):
        h, mean, sem = sim.regret_at(pol, [50, 200, 800])
        assert np.all(mean > 0)
