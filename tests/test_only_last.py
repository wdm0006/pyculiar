"""only_last filters the finished result without truncating detection context."""

import datetime

import numpy as np
import pandas as pd
import pytest

from pyculiar import detect_ts

START = 1_600_000_000 - (1_600_000_000 % 3600)
N = 24 * 21
LAST = N - 1
WINDOW = pd.Timedelta(hours=30)  # shorter than two 24-hour seasonal periods
CUTOFF_IDX = LAST - 30  # excluded boundary
BEFORE, AT, AFTER = 300, CUTOFF_IDX, 490


def _hourly(spikes=(BEFORE, AT, AFTER)):
    rng = np.random.default_rng(3)
    v = 100 + 10 * np.sin(2 * np.pi * np.arange(N) / 24) + rng.normal(0, 0.5, N)
    for i in spikes:
        v[i] += 90
    return pd.DataFrame({"t": START + 3600 * np.arange(N, dtype="int64"), "v": v})


def _call(df=None, **kw):
    df = _hourly() if df is None else df
    kw.setdefault("direction", "pos")
    return detect_ts(df, max_anoms=0.02, granularity="hr", inplace=False, **kw)["anoms"]


def _idx(res):
    return sorted(int((t - START) // 3600) for t in res.timestamp)


def test_precondition_full_history_detects_all_spikes():
    assert {BEFORE, AT, AFTER} <= set(_idx(_call()))


@pytest.mark.parametrize("e_value", [False, True])
@pytest.mark.parametrize("return_details", [False, True])
@pytest.mark.parametrize("longterm", [False, True])
def test_filter_equals_slice_of_full_result(e_value, return_details, longterm):
    kw = {"e_value": e_value, "return_details": return_details, "longterm": longterm}
    full = _call(**kw)
    assert {BEFORE, AT, AFTER} <= set(_idx(full))
    got = _call(only_last=WINDOW, **kw)
    cutoff = START + 3600 * CUTOFF_IDX
    expected = full[full.timestamp > cutoff]
    pd.testing.assert_frame_equal(got, expected)
    idx = _idx(got)
    assert AFTER in idx and BEFORE not in idx and AT not in idx
    assert list(got.columns) == list(full.columns)


def test_none_preserves_results():
    pd.testing.assert_frame_equal(_call(only_last=None), _call())


def test_latest_timestamp_included_and_anchor_is_input_not_clock():
    df = _hourly(spikes=(BEFORE, LAST))
    res = _call(df, only_last=pd.Timedelta(hours=1))
    assert _idx(res) == [LAST]
    # latest row not anomalous: anchor still the latest input time
    df = _hourly(spikes=(BEFORE, 495))
    res = _call(df, only_last=pd.Timedelta(hours=10))
    assert _idx(res) == [495]
    assert len(_call(df, only_last=pd.Timedelta(hours=1))) == 0


def test_boundary_excluded_and_one_second_wider_includes():
    res = _call(only_last=pd.Timedelta(hours=30) + pd.Timedelta(seconds=1))
    assert AT in _idx(res)


def test_unix_and_aware_datetimes_agree():
    base = _hourly()
    unix = _call(base, only_last=WINDOW, e_value=True, return_details=True)
    tz = datetime.timezone(datetime.timedelta(hours=-5))
    aware = base.copy()
    aware["t"] = pd.to_datetime(aware["t"], unit="s", utc=True).dt.tz_convert(tz)
    naive = base.copy()
    naive["t"] = pd.to_datetime(naive["t"], unit="s")
    for other in (aware, naive):
        res = _call(other, only_last=WINDOW, e_value=True, return_details=True)
        pd.testing.assert_frame_equal(res, unix)
    assert len(unix) > 0


@pytest.mark.parametrize("e_value", [False, True])
@pytest.mark.parametrize("return_details", [False, True])
def test_all_detections_filtered_keeps_columns(e_value, return_details):
    df = _hourly(spikes=(BEFORE,))
    full = _call(df, e_value=e_value, return_details=return_details)
    assert len(full) > 0
    res = _call(df, only_last=pd.Timedelta(hours=1), e_value=e_value, return_details=return_details)
    assert len(res) == 0
    assert list(res.columns) == list(full.columns)


def test_insufficient_context_unchanged():
    short = _hourly().iloc[:30].reset_index(drop=True)
    res = _call(short, only_last=pd.Timedelta(hours=1), e_value=True)
    assert len(res) == 0 and list(res.columns) == ["timestamp", "anoms"]


@pytest.mark.parametrize("inplace", [True, False])
@pytest.mark.parametrize(
    "bad",
    [
        pd.Timedelta(0),
        pd.Timedelta(hours=-1),
        pd.NaT,
        3600,
        1.5,
        "1h",
        datetime.timedelta(hours=1),
        np.timedelta64(1, "h"),
        True,
    ],
)
def test_invalid_only_last_rejected_before_mutation(bad, inplace):
    df = _hourly()
    before = df.copy()
    with pytest.raises(ValueError, match="only_last"):
        detect_ts(df, granularity="hr", inplace=inplace, only_last=bad)
    pd.testing.assert_frame_equal(df, before)


@pytest.mark.parametrize("inplace", [True, False])
def test_invalid_rejected_even_when_context_insufficient(inplace):
    short = _hourly().iloc[:30].reset_index(drop=True)
    with pytest.raises(ValueError, match="only_last"):
        detect_ts(short, granularity="hr", inplace=inplace, only_last=pd.Timedelta(0))
