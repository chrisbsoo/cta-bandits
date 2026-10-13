"""Named experiment configurations (what to run), kept separate from how to run them."""
import numpy as np
from .experiment import Experiment, PolicySpec

BASE_POLICIES = (PolicySpec.of("ucb1"), PolicySpec.of("etc", gamma=2 / 3))
ALL_POLICIES = (PolicySpec.of("ucb1"), PolicySpec.of("etc", gamma=2 / 3), PolicySpec.of("glrucb"),
                PolicySpec.of("swucb", tau=50), PolicySpec.of("rawucb"), PolicySpec.of("sswucb"), PolicySpec.of("greedy"))


def smoke():
    """Seconds, local: one cell per region and m, to check the pipeline end to end."""
    return Experiment.grid(alphas=(0.3, 0.5, 0.7), betas=(0.5,), ms=(1, 2), Ks=(3,), policies=BASE_POLICIES,
                           horizons=(100, 300, 1000), n_runs=20)


def regions(n_alpha=5, n_inst=5, n_runs=100):
    """(alpha, beta) grid x m in {1,2,3} x random instances; the grid includes the contested diagonal."""
    g = np.round(np.linspace(0.15, 0.85, n_alpha), 3)
    return Experiment.grid(alphas=g, betas=g, ms=(1, 2, 3), Ks=(4,), policies=BASE_POLICIES,
                           horizons=tuple(int(x) for x in np.geomspace(100, 5000, 8)),
                           n_runs=n_runs, instance_seeds=tuple(range(n_inst)))


def learners(n_runs=40, n_inst=2):
    """All seven learners on a coarse (alpha, beta) grid x m in {1,2,3}: the full-comparison preset."""
    g = (0.3, 0.5, 0.7)
    return Experiment.grid(alphas=g, betas=g, ms=(1, 2, 3), Ks=(4,), policies=ALL_POLICIES,
                           horizons=(100, 300, 1000, 3000), n_runs=n_runs, instance_seeds=tuple(range(n_inst)))

def grid10(n_alpha=5, n_inst=5, n_runs=40, K=10, policies=BASE_POLICIES):
    """K=10: (alpha, beta) grid x frontier size m = 1..K x random instances (average case per cell)."""
    g = np.round(np.linspace(0.15, 0.85, n_alpha), 3)
    return Experiment.grid(alphas=g, betas=g, ms=tuple(range(1, K + 1)), Ks=(K,), policies=policies,
                           horizons=(100, 300, 1000, 3000), n_runs=n_runs, instance_seeds=tuple(range(n_inst)))

def grid10_glr_sw(**kw):
    """Same K=10 grid for GLR-UCB and SW-UCB."""
    return grid10(policies=(PolicySpec.of("glrucb"), PolicySpec.of("swucb", tau=50)), **kw)

def nested10(n_alpha=5, n_inst=5, n_runs=40, K=10, ms=(1, 2, 5, 10), policies=BASE_POLICIES,
             horizons=(100, 300, 1000, 3000, 10000, 30000)):
    """Fixed-gap nested instances, K=10, a few frontier sizes, longer horizons."""
    g = np.round(np.linspace(0.15, 0.85, n_alpha), 3)
    return Experiment.grid(alphas=g, betas=g, ms=tuple(ms), Ks=(K,), policies=policies, horizons=horizons,
                           n_runs=n_runs, instance_seeds=tuple(range(n_inst)), kind="nested")

GLR_SW = (PolicySpec.of("glrucb"), PolicySpec.of("swucb", tau=50))


def nested10_glr_sw_pilot():
    """Small pilot of the Modal run (about a minute on a laptop): use it to check the pipeline end to end."""
    return nested10(n_alpha=3, n_inst=2, n_runs=20, policies=GLR_SW, horizons=(100, 300, 1000, 3000))


def nested10_glr_sw():
    """High-resolution Modal run for GLR-UCB and SW-UCB: 7x7 (alpha, beta) grid, K=10, m in {1,2,5,10},
    10 instances x 100 runs, horizons up to 30000. 7*7*4*10 = 1960 cells."""
    return nested10(n_alpha=7, n_inst=10, n_runs=100, policies=GLR_SW,
                    horizons=(100, 300, 1000, 3000, 10000, 30000))

def nested10_all():
    """ONE high-resolution run for every learner, so all figures share grid, instances, runs and horizons.
    UCB1, ETC, GLR-UCB, SW-UCB: 1960 cells (7x7 grid, m in {1,2,5,10}, 10 instances), 100 runs, T up to 30000."""
    return nested10(n_alpha=7, n_inst=10, n_runs=100, policies=BASE_POLICIES + GLR_SW,
                    horizons=(100, 300, 1000, 3000, 10000, 30000))

PRESETS = {"smoke": smoke, "regions": regions, "learners": learners,
           "grid10": grid10, "grid10_glr_sw": grid10_glr_sw, "nested10": nested10, "nested10_glr_sw_pilot": nested10_glr_sw_pilot, 
           "nested10_glr_sw": nested10_glr_sw, "nested10_all": nested10_all}

