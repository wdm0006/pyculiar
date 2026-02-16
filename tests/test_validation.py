"""Input validation tests for detect_ts and detect_anoms.

Tests all error paths to ensure consistent error handling after the C rewrite.
"""

import datetime
import os

import numpy as np
import pandas as pd
import pytest

from pyculiarity import detect_ts
from pyculiarity.detect_anoms import detect_anoms


@pytest.fixture
def raw_data():
    path = os.path.dirname(os.path.realpath(__file__))
    data = pd.read_csv(
        os.path.join(path, 'raw_data.csv'),
        usecols=['timestamp', 'count'])
    data['timestamp'] = pd.to_datetime(data['timestamp']).map(pd.Timestamp.timestamp).astype(int)
    return data


@pytest.fixture
def raw_data_df():
    """DataFrame in detect_anoms format (timestamp as datetime, value column)."""
    path = os.path.dirname(os.path.realpath(__file__))
    data = pd.read_csv(
        os.path.join(path, 'raw_data.csv'),
        usecols=['timestamp', 'count'])
    data['timestamp'] = pd.to_datetime(data['timestamp']).map(pd.Timestamp.timestamp).astype(int)
    data.columns = ['timestamp', 'value']
    data['timestamp'] = data['timestamp'].map(
        lambda x: datetime.datetime.fromtimestamp(x, tz=datetime.timezone.utc))
    return data


# ---------------------------------------------------------------------------
# detect_ts validation
# ---------------------------------------------------------------------------

class TestDetectTsInputValidation:
    def test_non_dataframe_input(self):
        with pytest.raises(ValueError, match="data must be a single data frame"):
            detect_ts([1, 2, 3])

    def test_wrong_number_of_columns(self):
        df = pd.DataFrame({'a': [1, 2], 'b': [3, 4], 'c': [5, 6]})
        with pytest.raises(ValueError):
            detect_ts(df)

    def test_single_column(self):
        df = pd.DataFrame({'a': [1, 2, 3]})
        with pytest.raises(ValueError):
            detect_ts(df)

    def test_non_numeric_values(self):
        df = pd.DataFrame({'timestamp': [1, 2, 3], 'value': ['a', 'b', 'c']})
        with pytest.raises(ValueError):
            detect_ts(df)

    def test_string_timestamps(self):
        df = pd.DataFrame({
            'timestamp': ['2020-01-01', '2020-01-02', '2020-01-03'],
            'value': [1.0, 2.0, 3.0]
        })
        with pytest.raises(ValueError):
            detect_ts(df)

    def test_max_anoms_too_high(self, raw_data):
        with pytest.raises(ValueError, match="max_anoms must be less than 50"):
            detect_ts(raw_data, max_anoms=0.50)

    def test_invalid_direction(self, raw_data):
        with pytest.raises(ValueError, match="direction options"):
            detect_ts(raw_data, direction='up')

    def test_invalid_threshold(self, raw_data):
        with pytest.raises(ValueError, match="threshold options"):
            detect_ts(raw_data, threshold='invalid')

    def test_e_value_not_bool(self, raw_data):
        with pytest.raises(ValueError, match="e_value must be a boolean"):
            detect_ts(raw_data, e_value='yes')

    def test_longterm_not_bool(self, raw_data):
        with pytest.raises(ValueError, match="longterm must be a boolean"):
            detect_ts(raw_data, longterm='yes')

    def test_piecewise_median_period_weeks_too_small(self, raw_data):
        with pytest.raises(ValueError, match="piecewise_median_period_weeks"):
            detect_ts(raw_data, piecewise_median_period_weeks=1)

    def test_unsupported_granularity(self, raw_data):
        with pytest.raises(ValueError, match="not supported"):
            detect_ts(raw_data, granularity='year')


# ---------------------------------------------------------------------------
# detect_anoms validation
# ---------------------------------------------------------------------------

class TestDetectAnomsInputValidation:
    def test_no_period(self, raw_data_df):
        with pytest.raises(ValueError, match="must supply period"):
            detect_anoms(raw_data_df, num_obs_per_period=None)

    def test_insufficient_data(self, raw_data_df):
        """Less than 2 periods of data should return None (prints warning)."""
        small = raw_data_df.head(100).copy()
        result = detect_anoms(small, k=0.02, alpha=0.05, num_obs_per_period=1440)
        assert result is None

    def test_internal_nan(self, raw_data_df):
        """Internal NaN values should raise ValueError."""
        df = raw_data_df.copy()
        mid = len(df) // 2
        df.at[mid, 'value'] = np.nan
        with pytest.raises(ValueError, match="non-leading NAs"):
            detect_anoms(df, k=0.02, alpha=0.05, num_obs_per_period=1440)
