import matplotlib
matplotlib.use("Agg")
from tacb.experiment import Experiment, LocalExecutor, Results
from tacb.presets import PRESETS
from tacb import plots


def _small():
    exp = PRESETS["smoke"]()
    return exp.run(LocalExecutor(1), chunk_size=4)


def test_csv_roundtrip(tmp_path):
    res = _small()
    p = tmp_path / "r.csv"
    res.to_csv(p)
    back = Results.from_csv(p)
    assert len(back.df) == len(res.df)
    assert set(back.df.columns) == set(res.df.columns)


def test_write_report_smoke(tmp_path):
    res = _small()
    plots.write_report(res, tmp_path)
    names = {f.name for f in tmp_path.iterdir()}
    assert any(n.endswith(".png") for n in names)
    assert any(n.endswith(".csv") for n in names)
