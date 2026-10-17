"""return_details=True adds per-anomaly direction and residual."""

import numpy as np
import pandas as pd
import pytest

from pyculiar import detect_ts
from pyculiar._cext.anomaly_module import seasonal_decompose

START = 1_600_000_000 - (1_600_000_000 % 3600)
SPIKE, DIP = 200, 300
BASE = ["timestamp", "anoms"]


def _hourly(n=24 * 21):
    rng = np.random.default_rng(1)
    v = 100 + 10 * np.sin(2 * np.pi * np.arange(n) / 24) + rng.normal(0, 0.5, n)
    v[SPIKE] += 80
    v[DIP] -= 80
    return pd.DataFrame({"t": START + 3600 * np.arange(n, dtype="int64"), "v": v})


def _call(df=None, **kw):
    df = _hourly() if df is None else df
    return detect_ts(df, max_anoms=0.01, direction="both", granularity="hr", inplace=False, **kw)["anoms"]


def test_default_columns_unchanged():
    assert list(_call().columns) == BASE
    assert list(_call(e_value=True).columns) == BASE + ["expected_value"]


def test_direction_by_value():
    res = _call(return_details=True).set_index("timestamp")
    assert res.loc[START + 3600 * SPIKE, "direction"] == "pos"
    assert res.loc[START + 3600 * DIP, "direction"] == "neg"


def test_residual_matches_independent_recomputation():
    df = _hourly()
    values = df.v.tolist()
    _, seasonal, _ = seasonal_decompose(values, 24)
    med = float(np.median(values))
    res = _call(df, return_details=True)
    assert {SPIKE, DIP} <= {int((t - START) // 3600) for t in res.timestamp}
    for _, row in res.iterrows():
        i = int((row.timestamp - START) // 3600)
        assert row.residual == pytest.approx(values[i] - seasonal[i] - med)
        assert row.direction == ("pos" if row.residual > 0 else "neg")


def test_longterm_threshold_and_e_value():
    res = _call(return_details=True, longterm=True, e_value=True)
    assert list(res.columns) == BASE + ["expected_value", "direction", "residual"]
    assert res.timestamp.is_unique
    d = res.set_index("timestamp").direction
    assert d[START + 3600 * SPIKE] == "pos" and d[START + 3600 * DIP] == "neg"
    th = _call(return_details=True, threshold="med_max")
    assert set(th.direction) <= {"pos", "neg"} and th.residual.notna().all()


def test_empty_result_has_columns():
    res = detect_ts(
        _hourly(), max_anoms=0.01, alpha=1e-12, direction="pos", granularity="hr", inplace=False, return_details=True
    )["anoms"]
    if len(res) == 0:
        assert {"direction", "residual"} <= set(res.columns)


@pytest.mark.parametrize("bad", [1, "yes", None])
def test_non_bool_rejected_before_mutation(bad):
    df = _hourly()
    before = df.copy()
    with pytest.raises(ValueError, match="return_details"):
        detect_ts(df, granularity="hr", return_details=bad)
    pd.testing.assert_frame_equal(df, before)
