import pickle
import numpy as np
from tacb.experiment import Experiment, PolicySpec, LocalExecutor, run_cells
from tacb.presets import smoke


def tiny():
    return Experiment.grid(alphas=(0.3, 0.6), betas=(0.5,), ms=(1, 2), Ks=(3,), instance_seeds=(0, 1),
                           policies=(PolicySpec.of("ucb1"), PolicySpec.of("etc", gamma=0.6)),
                           horizons=(50, 150, 450), n_runs=10)


def test_cells_are_picklable_and_deterministic():
    cell = tiny().cells[0]
    cell2 = pickle.loads(pickle.dumps(cell))
    assert cell == cell2 and cell.run() == cell2.run()


def test_results_table_shape_and_columns():
    res = tiny().run()
    assert len(res.df) == len(tiny().cells) * 2
    assert {"region", "m", "oracle", "policy", "exponent", "regret_mean"} <= set(res.df.columns)
    long = res.to_long()
    assert len(long) == len(res.df) * 3 and (long["regret_mean"] > 0).all()


def test_process_pool_matches_sequential():
    seq = tiny().run(LocalExecutor(1), chunk_size=2).df
    par = tiny().run(LocalExecutor(2), chunk_size=2).df
    assert seq[["region", "m", "policy", "exponent"]].equals(par[["region", "m", "policy", "exponent"]])


def test_modal_function_matches_local_and_executor_interface():
    from tacb import modal_app
    cells = tiny().cells[:3]
    assert modal_app.run_cells_remote.local(cells) == run_cells(cells)       # same code path as the container
    # ModalExecutor has the LocalExecutor interface: map(list of chunks) -> list of row-lists
    fake = lambda chunks: [run_cells(c) for c in chunks]
    ex = modal_app.ModalExecutor(); ex_map = modal_app.run_cells_remote.map
    try:
        modal_app.run_cells_remote.map = fake        # stand in for the remote call
        assert ex.map([cells[:1], cells[1:]]) == [run_cells(cells[:1]), run_cells(cells[1:])]
    finally:
        modal_app.run_cells_remote.map = ex_map


def test_smoke_preset_runs():
    assert len(smoke().cells) == 6
