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


def test_both_directions(raw_data):
    results = detect_ts(raw_data, max_anoms=0.02, direction="both", granularity="min")
    assert len(results["anoms"].columns) == 2
    assert len(results["anoms"].iloc[:, 1]) > 0


def test_both_directions_e_value_longterm(raw_data):
    results = detect_ts(raw_data, max_anoms=0.02, direction="both", longterm=True, e_value=True, granularity="min")
    assert len(results["anoms"].columns) == 3
    assert len(results["anoms"].iloc[:, 1]) > 0


def test_both_directions_e_value_threshold_med_max(raw_data):
    results = detect_ts(
        raw_data, max_anoms=0.02, direction="both", threshold="med_max", e_value=True, granularity="min"
    )
    assert len(results["anoms"].columns) == 3
    assert len(results["anoms"].iloc[:, 1]) > 0


@pytest.fixture
def overlapping_windows_data():
    rng = np.random.default_rng(1)
    n = 24 * 45
    timestamps = 1_699_999_200 + 3600 * np.arange(n)
    values = 100 + 20 * np.sin(2 * np.pi * np.arange(n) / 24) + rng.normal(0, 2, n)
    values[n - 20] += 80
    return pd.DataFrame({"timestamp": timestamps, "value": values})


@pytest.mark.parametrize("e_value", [False, True])
def test_longterm_overlapping_windows_have_unique_anoms(overlapping_windows_data, e_value):
    results = detect_ts(
        overlapping_windows_data, max_anoms=0.05, direction="pos", longterm=True, e_value=e_value, granularity="hr"
    )
    anoms = results["anoms"]
    assert anoms["timestamp"].is_unique
    assert len(anoms) == 2
    if e_value:
        assert anoms["expected_value"].notna().all()
