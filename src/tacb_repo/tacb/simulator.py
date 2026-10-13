"""The ONE Monte Carlo step, shared by every policy: select -> pull -> update."""
from dataclasses import dataclass
import numpy as np
from .env import Environment
from .model import Instance
from .oracle import Oracle
from .policies.base import Policy, Context


@dataclass
class RunResult:
    horizons: np.ndarray            # checkpoints (1-based rounds)
    cum_mean_reward: np.ndarray     # (R, H) cumulative TRUE mean reward at each checkpoint
    final_counts: np.ndarray        # (R, K) pulls per arm at the final round

    def regret(self, oracle: Oracle):
        """Pseudo-regret V*(h) - cumulative mean reward, shape (R, H)."""
        return oracle.values(self.horizons)[None, :] - self.cum_mean_reward


class Simulator:
    def __init__(self, instance: Instance, n_runs: int = 100, seed: int = 0):
        self.inst, self.R, self.seed = instance, int(n_runs), seed
        self.oracle = Oracle(instance)

    def run(self, policy: Policy, T: int, checkpoints=None) -> RunResult:
        T = int(T)
        cps = np.unique(np.asarray([T] if checkpoints is None else checkpoints, int))
        assert cps.min() >= 1 and cps.max() <= T
        env_rng, pol_rng = (np.random.default_rng(s) for s in np.random.SeedSequence(self.seed).spawn(2))
        env = Environment(self.inst, self.R, env_rng)
        policy.reset(Context(n_runs=self.R, K=self.inst.K, T=T, sigma=self.inst.sigma, rng=pol_rng,
                           alpha=self.inst.alpha, beta=self.inst.beta))
        cum = np.zeros(self.R)
        out = np.empty((self.R, len(cps)))
        nxt = 0
        for t in range(1, T + 1):
            arms = policy.select(t)
            rewards, mu = env.step(arms, t)
            policy.update(t, arms, rewards)
            cum += mu
            if t == cps[nxt]:
                out[:, nxt] = cum
                nxt += 1
        return RunResult(cps, out, env.n.copy())

    def regret_at(self, policy: Policy, horizons):
        """Mean regret and standard error at each horizon. Anytime policies run once with checkpoints;
        horizon-dependent policies (e.g. ETC) are rerun for every horizon."""
        horizons = np.asarray(sorted(set(int(h) for h in horizons)))
        if policy.anytime:
            reg = self.run(policy, horizons.max(), horizons).regret(self.oracle)
        else:
            reg = np.column_stack([self.run(policy, h).regret(self.oracle)[:, 0] for h in horizons])
        return horizons, reg.mean(axis=0), reg.std(axis=0, ddof=1) / np.sqrt(self.R)
