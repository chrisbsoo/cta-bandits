"""Modal integration. The SAME Experiment/Cell code runs here; only the executor changes.

    modal run -m tacb.modal_app --preset regions --chunk-size 4 --out regions.csv

Containers are ephemeral, so results are returned to the local entrypoint (rows, not files) and written
locally. Run from the repository root so that `tacb` is importable when the image is built.
"""
import modal

app = modal.App("tacb-experiments")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy", "numba", "scipy", "pandas")
    .env({"NUMBA_CACHE_DIR": "/tmp/numba_cache"})          # package dir may be read-only in the container
    .add_local_python_source("tacb")
)


@app.function(image=image, cpu=4.0, timeout=3600)
def run_cells_remote(cells: list) -> list:
    """Run one chunk of cells inside a container; returns result rows."""
    from tacb.experiment import run_cells
    return run_cells(cells)


class ModalExecutor:
    """Executor with the same interface as LocalExecutor. Needs an active app context:
    inside `modal run` this is automatic; from a plain script use `with app.run(): ...`."""

    def __init__(self, partial=None):
        self.partial = partial                 # optional CSV path, rewritten after every finished chunk

    def map(self, chunks):
        import time
        from tacb.experiment import tick, Results
        import pandas as pd
        t0, out = time.time(), []
        for r in run_cells_remote.map(chunks):
            out.append(r); tick(len(out), len(chunks), t0)
            if self.partial:
                Results(pd.DataFrame([row for part in out for row in part])).to_csv(self.partial)
        return out


@app.local_entrypoint()
def main(preset: str = "smoke", chunk_size: int = 4, out: str = ""):
    from tacb.presets import PRESETS
    exp = PRESETS[preset]()
    print(f"Dispatching {len(exp.cells)} cells in chunks of {chunk_size} to Modal...")
    res = exp.run(ModalExecutor(partial=(out + ".partial.csv") if out else None), chunk_size)
    print(res.exponents().to_string(index=False))
    if out:
        res.to_csv(out)
        print("saved", out)
