import numpy as np
import pandas as pd
import pytest

from pyculiar import detect_ts

START = 1_600_000_000 - (1_600_000_000 % 3600)
SPIKE_ROW = 100


def _series():
    n = 24 * 14
    rng = np.random.default_rng(3)
    v = 100 + 10 * np.sin(2 * np.pi * np.arange(n) / 24) + rng.normal(0, 0.5, n)
    v[SPIKE_ROW] += 50
    return pd.DataFrame({"ts": START + 3600 * np.arange(n), "val": v})


def test_series_without_inf_detects_spike():
    res = detect_ts(_series(), granularity="hr", inplace=False)["anoms"]
    assert (START + 3600 * SPIKE_ROW) in set(res["timestamp"])


@pytest.mark.parametrize("bad", [np.inf, -np.inf])
@pytest.mark.parametrize("inplace", [True, False])
def test_inf_raises_and_preserves_columns(bad, inplace):
    df = _series()
    df.loc[200, "val"] = bad
    with pytest.raises(ValueError, match="infinite"):
        detect_ts(df, granularity="hr", inplace=inplace)
    assert list(df.columns) == ["ts", "val"]
