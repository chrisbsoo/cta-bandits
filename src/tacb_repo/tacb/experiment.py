"""Experiments: picklable work units (Cell), grid construction, executors, tidy results.

A Cell is the unit of parallel work: ONE instance x a list of policies. Policies in a cell share the
instance's oracle (the block DP is the expensive part). Cells are tiny, picklable recipes, so the same
Experiment runs unchanged locally or on Modal: only the executor differs.
"""
from dataclasses import dataclass, field, asdict
from itertools import product
import numpy as np
import pandas as pd
from .analysis import fit_exponent
from .generators import make_instance, make_nested
from .policies.registry import make_policy
from .simulator import Simulator


@dataclass(frozen=True)
class InstanceSpec:
    """Recipe for a random instance with exactly m frontier arms."""
    K: int
    m: int
    alpha: float
    beta: float
    seed: int = 0
    sigma: float = 1.0
    kind: str = "random"

    def build(self):
        return make_instance(self.K, self.m, self.alpha, self.beta, rng=self.seed, sigma=self.sigma)


@dataclass(frozen=True)
class PolicySpec:
    name: str
    params: tuple = ()           # sorted ((key, value), ...): hashable and picklable

    @staticmethod
    def of(name, **params):
        return PolicySpec(name, tuple(sorted(params.items())))

    def build(self):
        return make_policy(self.name, **dict(self.params))

    @property
    def label(self):
        return self.name if not self.params else f"{self.name}(" + ",".join(f"{k}={v:g}" if isinstance(v, float) else f"{k}={v}" for k, v in self.params) + ")"


@dataclass(frozen=True)
class Cell:
    instance: InstanceSpec
    policies: tuple              # tuple[PolicySpec]
    horizons: tuple
    n_runs: int = 100
    seed: int = 0

    def run(self):
        """Execute the cell and return one result row (dict) per policy."""
        inst = self.instance.build()
        sim = Simulator(inst, self.n_runs, self.seed)
        rows = []
        for ps in self.policies:
            h, mean, sem = sim.regret_at(ps.build(), self.horizons)
            e, c, r2, se = fit_exponent(h, mean)
            rows.append(dict(
                K=inst.K, m=inst.m, alpha=inst.alpha, beta=inst.beta, region=inst.region,
                oracle=sim.oracle.method, inst_seed=self.instance.seed, policy=ps.label,
                n_runs=self.n_runs, exponent=e, const=c, r2=r2, exp_se=se,
                v_star=float(sim.oracle.solve(int(h[-1])).value), kind=self.instance.kind,
                horizons=[int(x) for x in h], regret_mean=[float(x) for x in mean],
                regret_sem=[float(x) for x in sem]))
        return rows


def run_cell(cell: Cell):
    """Module-level so executors (process pools, Modal) can pickle/reference it."""
    return cell.run()


def run_cells(cells):
    """Run a chunk of cells; return a flat list of rows."""
    return [row for cell in cells for row in cell.run()]

def tick(done, total, t0):
    """One-line progress with elapsed time and a rough ETA (cells differ in cost, so the ETA is approximate)."""
    import sys, time
    el = time.time() - t0
    eta = el / done * (total - done)
    print(f"\r[{done}/{total} chunks] {100 * done / total:5.1f}%  elapsed {el / 60:6.1f} min  eta {eta / 60:6.1f} min",
          end="\n" if done == total else "", file=sys.stderr, flush=True)


# ------------------------------------------------------------------------------------------ executors
class LocalExecutor:
    """Sequential by default; n_jobs > 1 uses a process pool (each worker limited to 1 numba thread)."""

    def __init__(self, n_jobs: int = 1):
        self.n_jobs = n_jobs

    def map(self, chunks):
        import time
        t0, total = time.time(), len(chunks)
        if self.n_jobs <= 1:
            out = []
            for c in chunks:
                out.append(run_cells(c)); tick(len(out), total, t0)
            return out
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor, as_completed
        # 'spawn', not 'fork': forking after numba's parallel threads have started can crash the child
        with ProcessPoolExecutor(self.n_jobs, mp_context=mp.get_context("spawn"), initializer=_limit_threads) as pool:
            futs = {pool.submit(run_cells, c): i for i, c in enumerate(chunks)}
            out = [None] * total
            for k, f in enumerate(as_completed(futs), 1):
                out[futs[f]] = f.result(); tick(k, total, t0)
            return out


def _limit_threads():
    import numba
    numba.set_num_threads(1)


# ------------------------------------------------------------------------------------------ experiment
@dataclass
class Results:
    df: pd.DataFrame

    def to_csv(self, path):
        out = self.df.copy()
        for col in ("horizons", "regret_mean", "regret_sem"):      # lists -> JSON-ish strings for CSV
            out[col] = out[col].map(lambda v: " ".join(f"{x:.6g}" for x in v))
        out.to_csv(path, index=False)

    @classmethod
    def from_csv(cls, path):
        df = pd.read_csv(path)
        for col in ("horizons", "regret_mean", "regret_sem"):
            conv = int if col == "horizons" else float
            df[col] = df[col].map(lambda v: [conv(float(x)) for x in str(v).split()])
        return cls(df)

    def to_long(self):
        """One row per (cell, policy, horizon)."""
        id_cols = [c for c in self.df.columns if c not in ("horizons", "regret_mean", "regret_sem")]
        return (self.df.explode(["horizons", "regret_mean", "regret_sem"])
                .rename(columns={"horizons": "T"}).astype({"T": int, "regret_mean": float, "regret_sem": float}))

    def exponents(self, by=("region", "m", "policy")):
        return self.df.groupby(list(by))["exponent"].agg(["mean", "std", "count"]).reset_index()


@dataclass
class Experiment:
    cells: list = field(default_factory=list)

    @classmethod
    def grid(cls, alphas, betas, ms, Ks=(4,), policies=(), horizons=(100, 300, 1000, 3000),
             n_runs=100, instance_seeds=(0,), sigma=1.0, seed=0, kind="random"):
        policies = tuple(policies)
        cells = [Cell(InstanceSpec(K, m, float(a), float(b), s, sigma, kind), policies, tuple(horizons), n_runs, seed)
                 for K, m, a, b, s in product(Ks, ms, alphas, betas, instance_seeds) if m <= K]
        return cls(cells)

    def chunked(self, chunk_size):
        return [self.cells[i:i + chunk_size] for i in range(0, len(self.cells), chunk_size)]

    def run(self, executor=None, chunk_size=4) -> Results:
        executor = executor or LocalExecutor()
        rows = [row for part in executor.map(self.chunked(chunk_size)) for row in part]
        return Results(pd.DataFrame(rows))
