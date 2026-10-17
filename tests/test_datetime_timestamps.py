"""detect_ts accepts datetime64 timestamp columns (naive = UTC, tz-aware -> UTC)."""

import os

import pandas as pd
import pytest

from pyculiar import detect_ts

HERE = os.path.dirname(os.path.realpath(__file__))


def _load(name, cols):
    data = pd.read_csv(os.path.join(HERE, name), usecols=cols)
    data[cols[0]] = pd.to_datetime(data[cols[0]]).map(pd.Timestamp.timestamp).astype(int)
    return data


def _as_datetime(data, tz):
    out = data.copy()
    col = out.columns[0]
    stamps = pd.to_datetime(out[col], unit="s")
    if tz:
        stamps = stamps.dt.tz_localize("UTC").dt.tz_convert(tz)
    out[col] = stamps
    return out


FIXTURES = {
    "raw_min": ("raw_data.csv", ["timestamp", "count"], {"max_anoms": 0.02, "direction": "both", "granularity": "min"}),
    "midnight_hr": (
        "midnight_test_data.csv",
        ["date", "value"],
        {"max_anoms": 0.05, "granularity": "hr", "e_value": True},
    ),
}


@pytest.mark.parametrize("tz", [None, "UTC", "US/Pacific", "Asia/Kolkata"])
@pytest.mark.parametrize("fixture", sorted(FIXTURES))
def test_datetime_matches_unix_seconds(fixture, tz):
    name, cols, kwargs = FIXTURES[fixture]
    unix = _load(name, cols)
    expected = detect_ts(unix.copy(), inplace=False, **kwargs)["anoms"]
    assert len(expected) > 0
    got = detect_ts(_as_datetime(unix, tz), inplace=False, **kwargs)["anoms"]
    pd.testing.assert_frame_equal(got.reset_index(drop=True), expected.reset_index(drop=True))
    assert got.timestamp.tolist() == expected.timestamp.tolist()


def test_naive_is_treated_as_utc():
    name, cols, kwargs = FIXTURES["raw_min"]
    unix = _load(name, cols)
    naive = detect_ts(_as_datetime(unix, None), inplace=False, **kwargs)["anoms"]
    shifted = _as_datetime(unix, "US/Pacific")
    shifted[cols[0]] = shifted[cols[0]].dt.tz_localize(None)  # local wall-clock, still read as UTC
    other = detect_ts(shifted, inplace=False, **kwargs)["anoms"]
    assert naive.timestamp.tolist() != other.timestamp.tolist()


@pytest.mark.parametrize("tz", [None, "US/Pacific"])
def test_inplace_false_leaves_frame_untouched(tz):
    name, cols, kwargs = FIXTURES["raw_min"]
    df = _as_datetime(_load(name, cols), tz)
    before = df.copy()
    detect_ts(df, inplace=False, **kwargs)
    pd.testing.assert_frame_equal(df, before)


def test_inplace_true_state():
    name, cols, kwargs = FIXTURES["raw_min"]
    unix = _load(name, cols)
    df = _as_datetime(unix, "US/Pacific")
    original = df.copy()
    detect_ts(df, inplace=True, **kwargs)
    assert list(df.columns) == cols
    assert df[cols[1]].tolist() == original[cols[1]].tolist()
    assert len(df) == len(original)
    # Timestamps keep denoting the same instants (now UTC python datetimes).
    assert [pd.Timestamp(t) for t in df[cols[0]]] == list(original[cols[0]])


@pytest.mark.parametrize("inplace", [True, False])
def test_validation_error_does_not_mutate(inplace):
    name, cols, _ = FIXTURES["raw_min"]
    df = _as_datetime(_load(name, cols), None).rename(columns={cols[0]: "when", cols[1]: "n"})
    before = df.copy()
    with pytest.raises(ValueError):
        detect_ts(df, max_anoms=0.5, inplace=inplace)
    pd.testing.assert_frame_equal(df, before)


def test_nat_rejected_without_mutation():
    df = pd.DataFrame({"t": pd.to_datetime(["2020-01-01", None, "2020-01-03"]), "v": [1.0, 2.0, 3.0]})
    before = df.copy()
    with pytest.raises(ValueError, match="NaT"):
        detect_ts(df)
    pd.testing.assert_frame_equal(df, before)


def test_string_timestamps_still_rejected():
    df = pd.DataFrame({"t": ["2020-01-01", "2020-01-02"], "v": [1.0, 2.0]})
    with pytest.raises(ValueError):
        detect_ts(df)
