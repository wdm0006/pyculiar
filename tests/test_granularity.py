"""Granularity support: 'sec' works end-to-end, 'ms' is rejected up front."""

import numpy as np
import pandas as pd
import pytest

from pyculiar import detect_ts

START = 1_600_000_000  # aligned to a whole second
SPIKE_OFFSETS = (5000, 9000)


def _per_second(n=14400):
    rng = np.random.default_rng(0)
    t = START + np.arange(n)
    v = 100 + 10 * np.sin(2 * np.pi * np.arange(n) / 3600) + rng.normal(0, 0.5, n)
    for off in SPIKE_OFFSETS:
        if off < n:
            v[off] += 80
    return pd.DataFrame({"timestamp": t.astype("int64"), "value": v})


@pytest.mark.parametrize("n", [10, 3000, 14400])
def test_ms_rejected_before_mutation(n):
    df = _per_second(n).rename(columns={"timestamp": "ts", "value": "val"})
    before = df.copy()
    with pytest.raises(ValueError, match="Supported values: sec, min, hr, day"):
        detect_ts(df, granularity="ms")
    pd.testing.assert_frame_equal(df, before)


def test_sec_detects_exact_spikes():
    df = _per_second()
    res = detect_ts(df, max_anoms=0.01, direction="pos", granularity="sec")["anoms"]
    assert sorted(res["timestamp"].astype("int64")) == [START + o for o in SPIKE_OFFSETS]
    assert sorted(res["anoms"].round(0)) == sorted(round(float(_per_second().value[o]), 0) for o in SPIKE_OFFSETS)


def test_sec_short_input_returns_empty_not_error():
    res = detect_ts(_per_second(3000), granularity="sec")["anoms"]
    assert len(res) == 0
