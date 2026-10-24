import os

import numpy as np
import pandas as pd
import pytest

from pyculiar import detect_ts


@pytest.fixture
def raw_data():
    path = os.path.dirname(os.path.realpath(__file__))
    data = pd.read_csv(os.path.join(path, "raw_data.csv"), usecols=["timestamp", "count"])
    data["timestamp"] = pd.to_datetime(data["timestamp"]).map(pd.Timestamp.timestamp).astype(int)
    return data


def test_handling_of_leading_trailing_nas(raw_data):
    for i in list(range(10)) + [len(raw_data) - 1]:
        raw_data.at[i, "count"] = np.nan

    results = detect_ts(raw_data, max_anoms=0.02, direction="both", granularity="min")
    assert len(results["anoms"].columns) == 2
    assert len(results["anoms"].iloc[:, 1]) > 0


def test_handling_of_middle_nas(raw_data):
    raw_data.at[len(raw_data) // 2, "count"] = np.nan
    with pytest.raises(ValueError):
        detect_ts(raw_data, max_anoms=0.02, direction="both", granularity="min")


START = 1_600_000_000 - (1_600_000_000 % 3600)
SPIKES = (200, 300)


def _gappy(gap):
    n = 24 * 21
    rng = np.random.default_rng(1)
    v = 100 + 10 * np.sin(2 * np.pi * np.arange(n) / 24) + rng.normal(0, 0.5, n)
    for s in SPIKES:
        v[s] += 80
    for i in gap:
        v[i] = np.nan
    return pd.DataFrame({"t": START + 3600 * np.arange(n, dtype="int64"), "v": v})


def _run(df, max_anoms=0.01, **kw):
    return detect_ts(df, max_anoms=max_anoms, direction="pos", granularity="hr", inplace=False, **kw)["anoms"]


def test_fill_missing_default_still_raises_without_zoo_reference():
    with pytest.raises(ValueError, match="fill_missing") as exc:
        _run(_gappy(range(100, 105)))
    assert "Zoo" not in str(exc.value)


def test_fill_missing_linear_finds_exact_spikes():
    res = _run(_gappy(range(100, 105)), fill_missing="linear")
    assert sorted(res.timestamp) == sorted(START + 3600 * s for s in SPIKES)


def test_fill_missing_never_reports_nan_timestamps():
    gap = range(201, 206)
    df = _gappy(gap)
    nan_ts = {START + 3600 * i for i in gap}
    res = _run(df, fill_missing="linear", max_anoms=0.02)
    assert START + 3600 * 200 in set(res.timestamp)
    assert not nan_ts & set(res.timestamp)
    assert START + 3600 * 300 in set(res.timestamp)
    assert res.anoms.notna().all()


def test_fill_missing_with_edge_nans():
    df = _gappy(list(range(5)) + list(range(100, 103)) + [24 * 21 - 1])
    res = _run(df, fill_missing="linear")
    assert sorted(res.timestamp) == sorted(START + 3600 * s for s in SPIKES)


@pytest.mark.parametrize("bad", ["cubic", "", True, 1, ["linear"]])
@pytest.mark.parametrize("inplace", [True, False])
def test_invalid_fill_missing_raises_before_mutation(bad, inplace):
    df = _gappy(range(100, 105))
    before = df.copy()
    with pytest.raises(ValueError, match="fill_missing"):
        detect_ts(df, granularity="hr", inplace=inplace, fill_missing=bad)
    assert list(df.columns) == ["t", "v"]
    pd.testing.assert_frame_equal(df, before)
