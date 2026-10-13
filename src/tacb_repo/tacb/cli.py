"""Command line.

    python -m tacb.cli run  --preset smoke [--n-jobs 2] [--out results.csv]
    python -m tacb.cli plot results.csv --out figures/
    modal run -m tacb.modal_app --preset regions --out results.csv        # then `plot` locally
"""
import argparse
from .experiment import LocalExecutor, Results
from .presets import PRESETS


def main(argv=None):
    p = argparse.ArgumentParser(prog="tacb")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run a preset locally")
    r.add_argument("--preset", default="smoke", choices=sorted(PRESETS))
    r.add_argument("--n-jobs", type=int, default=1)
    r.add_argument("--chunk-size", type=int, default=4)
    r.add_argument("--out", default=None)
    g = sub.add_parser("plot", help="make figures and tables from a results CSV")
    g.add_argument("csv")
    g.add_argument("--out", default="figures")
    a = p.parse_args(argv)
    if a.cmd == "run":
        exp = PRESETS[a.preset]()
        print(f"{len(exp.cells)} cells, local executor (n_jobs={a.n_jobs})")
        res = exp.run(LocalExecutor(a.n_jobs), a.chunk_size)
        print(res.exponents().to_string(index=False))
        if a.out:
            res.to_csv(a.out); print("saved", a.out)
    else:
        from .plots import write_report
        for path in write_report(Results.from_csv(a.csv), a.out):
            print("wrote", path)


if __name__ == "__main__":
    main()
