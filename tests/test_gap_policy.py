import numpy as np
import pandas as pd
import pytest

from pyculiar import detect_ts

SPIKE_IDX = 400


def make_hourly(offset=0, drop=None):
    n = 24 * 21
    ts = np.arange(n, dtype="int64") * 3600 + offset
    vals = 10 + 5 * np.sin(2 * np.pi * np.arange(n) / 24) + np.random.default_rng(0).normal(0, 0.1, n)
    vals[SPIKE_IDX] += 40
    df = pd.DataFrame({"t": ts, "v": vals})
    if drop is not None:
        df = df.drop(drop).reset_index(drop=True)
    return df


def test_error_names_first_gap():
    df = make_hourly(drop=range(100, 112))
    with pytest.raises(ValueError, match="first gap at 1970-01-05 04:00:00"):
        detect_ts(df, granularity="hr", gap_policy="error", inplace=False)


def test_interpolate_reports_spike_not_filled():
    df = make_hourly(drop=range(100, 112))
    res = detect_ts(df, granularity="hr", gap_policy="interpolate", inplace=False)["anoms"]
    assert list(res["timestamp"]) == [SPIKE_IDX * 3600.0]
    filled = {float(h * 3600) for h in range(100, 112)}
    assert not filled & set(res["timestamp"])


def test_default_with_gap_still_runs():
    df = make_hourly(drop=range(100, 112))
    detect_ts(df, granularity="hr", inplace=False)


def test_non_aligned_returns_original_row():
    df = make_hourly(offset=1800)
    res = detect_ts(df, granularity="hr", inplace=False)["anoms"]
    assert list(res["timestamp"]) == [SPIKE_IDX * 3600.0 + 1800]
    assert res["anoms"].iloc[0] == pytest.approx(df.v[SPIKE_IDX])


def test_error_policy_without_gaps_matches_default():
    df = make_hourly()
    a = detect_ts(df, granularity="hr", inplace=False)["anoms"]
    b = detect_ts(df, granularity="hr", gap_policy="error", inplace=False)["anoms"]
    pd.testing.assert_frame_equal(a, b)


@pytest.mark.parametrize("bad", ["fill", True, "", 1])
def test_invalid_gap_policy_raises_before_mutation(bad):
    df = make_hourly()
    cols = list(df.columns)
    with pytest.raises(ValueError, match="gap_policy"):
        detect_ts(df, granularity="hr", gap_policy=bad)
    assert list(df.columns) == cols
    assert df["t"].dtype == np.int64
