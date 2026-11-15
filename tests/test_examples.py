import importlib.util
from pathlib import Path

import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def _load(name):
    spec = importlib.util.spec_from_file_location(f"_example_{name}", EXAMPLES / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name", ["example", "example_2"])
def test_example_runs_headless_from_other_cwd(name, tmp_path, monkeypatch):
    cwd = tmp_path / "elsewhere"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    out = tmp_path / "out"

    results, plot_path = _load(name).main(output_dir=out)

    assert plot_path.parent == out
    assert plot_path.stat().st_size > 0
    assert len(results["anoms"]) > 0
    assert list(cwd.iterdir()) == []
