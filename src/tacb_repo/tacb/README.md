# tacb: Two-Clock Sigmoid Bandits

Independent arms, general K, alpha and beta in (0,1):  mu_i(t, n) = L_i + (U_i - L_i) * S(n**alpha - t**beta).

## Layout

| Module | Role |
|---|---|
| `model.Instance` | immutable problem instance (L, U, alpha, beta, sigma); region, frontier, m, mean reward |
| `frontier.find_frontier` | Algorithm 1, O(K log K) |
| `oracle.Oracle` | V*(T) by the cheapest exact method (dominance / no-switching / block DP), cached |
| `_kernels` | numba **pure functions** only (prefix sums, block DP); classes call these |
| `env.Environment` | true state (pull counts) + noise, vectorised over runs |
| `policies.*` | `Policy` interface; `UCB1`, `ETC`, `Greedy`, `SWUCB`, `RAWUCB`, `SSWUCB`, `GLRUCB`, `FixedSequence`, `OraclePolicy` |
| `simulator.Simulator` | the single Monte Carlo step shared by all policies; regret vs the oracle |
| `generators.make_instance` | random instances with a controlled frontier size m |
| `analysis` | regret curves and log-log exponent fits |

## One Monte Carlo step for every algorithm

```
for t in 1..T:   arms = policy.select(t)  ->  rewards, means = env.step(arms, t)  ->  policy.update(t, arms, rewards)
```
All independent runs advance together as arrays of shape (n_runs, K), so there is no per-run Python loop and
no numba inside policies. Policies see only rewards and the noise level sigma, never (L, U, alpha, beta).

## Adding a policy

Subclass `Policy`, implement `reset(ctx)`, `select(t)` (integer array of shape `(n_runs,)`) and
`update(t, arms, rewards)`. Set `anytime = False` if behaviour depends on the horizon (the simulator then reruns it
for each horizon). `StatsMixin` gives per-arm counts and sums.

## Tests

`python3 -m pytest tests -q`. The key end-to-end test: `OraclePolicy` replays the oracle's sequence and must have
exactly zero regret in every region; the oracle itself is checked against the Theta(T^K) reference DP.

## Experiments and Modal

```
python3 -m tacb.cli run --preset smoke --n-jobs 2 --out smoke.csv   # local
python3 -m tacb.cli plot smoke.csv --out figs                       # figures + companion CSV
modal run -m tacb.modal_app --preset regions --out regions.csv     # Modal, from the repository root
```
* `InstanceSpec` / `PolicySpec` / `Cell`: tiny picklable recipes; a `Cell` is one instance x several policies
  (policies share the instance's oracle, which is the expensive part).
* `Experiment.grid(...)` builds the cells; `Experiment.run(executor)` returns a `Results` table
  (`.exponents()`, `.to_long()`, `.to_csv()`).
* Executors share one interface, `map(chunks) -> list of row-lists`: `LocalExecutor(n_jobs)` (process pool uses
  `spawn`) and `modal_app.ModalExecutor` (fans chunks out to containers, returns rows, nothing is written remotely).
* `presets.py` holds *what* to run; add a preset there, no other file changes. New policies register themselves by name
  with `@register("name")` in `policies/registry.py` and can then be used in any `PolicySpec`.

## Plots

`plots.write_report(results, outdir)` writes `regret_grid.png` (regret vs T, region x m), `region_summary.png`
(fitted exponents), one exponent heatmap per policy, and a companion CSV with the plotted numbers. Policy colours
are bound to the policy name (fixed slot order), so filtering does not repaint. Identical curves overlap; use the CSV.

## Notes and caveats

* UCB-type bonuses are scaled by sigma: `c * sigma * sqrt(2 log(t+2) / n)`. The legacy code used an absolute bonus
  with sigma = 0.1, so reproducing its exploration needs a larger `ucb_c`.
* `GLRUCB` uses the known exponents (passed through `Context`), its commit test is only valid for alpha >= beta, and
  under its UCB fallback the losing arm can be starved, which keeps its variance estimate large.
* Modal has not been run for real here: try `--preset smoke` on Modal first. `.local()` parity with the local path is tested.
* Old scripts in the parent directory (`ucb.py`, `rho_grid.py`, ...) are superseded by this package and left untouched.
