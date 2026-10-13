"""Numeric summary tables from a results CSV.

    python -m tacb.summary all.csv --out tables/

Writes, and prints, four tables (CSV + a LaTeX tabular for each):
  rel_regret     regret per round R/T at every horizon (does it shrink with T?) and regret/V*(Tmax), by policy x region x m
  exponent       mean fitted exponent and the number of informative cells (regret >= 2% of V*)
  local_exponent slope between the last two horizons (does the regret keep growing at the end?)
  winners        how often each policy has the lowest relative regret at T_max, by region x m

Groups pool the (alpha, beta) grid points of a region and all instances, so "se" is the standard error across
those cells: it measures heterogeneity between cells, not just Monte Carlo noise.
"""
import argparse, os
import numpy as np
import pandas as pd
from .experiment import Results

REL_MIN = 0.02


def _prep(df):
    d = df.copy()
    d["rel_T"] = d.apply(lambda r: r["regret_mean"][-1] / r["v_star"], axis=1)
    d["local_exp"] = d.apply(lambda r: _local(r["horizons"], r["regret_mean"]), axis=1)
    d["informative"] = d["rel_T"] >= REL_MIN
    return d


def _local(h, reg):
    if len(h) < 2 or reg[-1] <= 0 or reg[-2] <= 0:
        return np.nan
    return float(np.log(reg[-1] / reg[-2]) / np.log(h[-1] / h[-2]))


def _ms(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, 0
    return x.mean(), (x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan), len(x)


def tables(df):
    d = _prep(df)
    keys = ["policy", "region", "m"]
    out = {}
    h = d["horizons"].iloc[0]
    rows = []
    for k, g in d.groupby(keys):
        row = dict(zip(keys, k)); row["cells"] = len(g)
        for i, T in enumerate(h):
            row[f"R/T@{T}"] = np.mean([r[i] / T for r in g["regret_mean"]])        # regret per round at horizon T
        row["rel@Tmax"] = g["rel_T"].mean()                                         # regret / V*(Tmax)
        row["se_rel@Tmax"] = _ms(g["rel_T"])[1]
        rows.append(row)
    out["rel_regret"] = pd.DataFrame(rows)
    rows = []
    for k, g in d.groupby(keys):
        gi = g[g["informative"]]
        m, se, n = _ms(gi["exponent"])
        rows.append(dict(zip(keys, k), exponent=m, se=se, informative_cells=n, cells=len(g)))
    out["exponent"] = pd.DataFrame(rows)
    rows = []
    for k, g in d.groupby(keys):
        gi = g[g["informative"]]
        m, se, n = _ms(gi["local_exp"])
        rows.append(dict(zip(keys, k), local_exponent_last=m, se=se, informative_cells=n))
    out["local_exponent"] = pd.DataFrame(rows)
    # winners: per (region, m, cell) pick the policy with the lowest rel_T
    cellkey = ["region", "m", "alpha", "beta", "inst_seed"]
    best = d.loc[d.groupby(cellkey)["rel_T"].idxmin()]
    w = best.groupby(["region", "m", "policy"]).size().rename("wins").reset_index()
    tot = d.groupby(["region", "m"])[cellkey[2:]].apply(lambda g: len(g.drop_duplicates())).rename("cells").reset_index()
    w = w.merge(tot, on=["region", "m"]); w["share"] = w["wins"] / w["cells"]
    out["winners"] = w
    return out


def _latex(df, caption=""):
    cols = list(df.columns)
    fmt = lambda v: (f"{v:.3f}" if isinstance(v, (float, np.floating)) else str(v).replace("_", r"\_"))
    lines = [r"\begin{tabular}{" + "l" * len(cols) + "}", r"\toprule",
             " & ".join(c.replace("_", r"\_").replace("@", r"@") for c in cols) + r" \\", r"\midrule"]
    lines += [" & ".join(fmt(v) for v in r) + r" \\" for r in df.itertuples(index=False)]
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("csv"); p.add_argument("--out", default="tables")
    a = p.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    df = Results.from_csv(a.csv).df
    if "v_star" not in df.columns:
        raise SystemExit("this CSV has no v_star column: rerun with the current code (nested10_* presets)")
    for name, t in tables(df).items():
        t = t.sort_values([c for c in ("region", "m", "policy") if c in t.columns]).reset_index(drop=True)
        t.to_csv(os.path.join(a.out, f"{name}.csv"), index=False)
        open(os.path.join(a.out, f"{name}.tex"), "w").write(_latex(t) + "\n")
        print(f"\n== {name} ==")
        print(t.round(3).to_string(index=False))
    print(f"\nwrote CSV and LaTeX tabulars to {a.out}/")


if __name__ == "__main__":
    main()