"""Golden tests for detect_anoms() intermediate function.

Tests detect_anoms() directly (bypassing detect_ts orchestration) to lock in
exact behavior before the C rewrite.
"""

import datetime
import os

import numpy as np
import pandas as pd
import pytest

from pyculiar.detect_anoms import detect_anoms


@pytest.fixture
def raw_data_df():
    """Load raw_data.csv and convert to the format detect_anoms expects."""
    path = os.path.dirname(os.path.realpath(__file__))
    data = pd.read_csv(
        os.path.join(path, 'raw_data.csv'),
        usecols=['timestamp', 'count'])
    data['timestamp'] = pd.to_datetime(data['timestamp']).map(pd.Timestamp.timestamp).astype(int)
    data.columns = ['timestamp', 'value']
    data['timestamp'] = data['timestamp'].map(
        lambda x: datetime.datetime.fromtimestamp(x, tz=datetime.timezone.utc))
    return data


class TestDetectAnomsBothDirections:
    """one_tail=False, upper_tail=True (maps to direction='both')"""

    def test_returns_dict(self, raw_data_df):
        r = detect_anoms(raw_data_df.copy(), k=0.02, alpha=0.05,
                         num_obs_per_period=1440, one_tail=False, upper_tail=True)
        assert isinstance(r, dict)
        assert 'anoms' in r
        assert 'stl' in r

    def test_anomaly_count(self, raw_data_df):
        r = detect_anoms(raw_data_df.copy(), k=0.02, alpha=0.05,
                         num_obs_per_period=1440, one_tail=False, upper_tail=True)
        assert len(r['anoms']) == 114

    def test_stl_shape(self, raw_data_df):
        r = detect_anoms(raw_data_df.copy(), k=0.02, alpha=0.05,
                         num_obs_per_period=1440, one_tail=False, upper_tail=True)
        assert r['stl'].shape == (14398, 2)

    def test_stl_columns(self, raw_data_df):
        r = detect_anoms(raw_data_df.copy(), k=0.02, alpha=0.05,
                         num_obs_per_period=1440, one_tail=False, upper_tail=True)
        assert list(r['stl'].columns) == ['timestamp', 'value']


class TestDetectAnomsPosDirection:
    """one_tail=True, upper_tail=True (maps to direction='pos')"""

    def test_anomaly_count(self, raw_data_df):
        r = detect_anoms(raw_data_df.copy(), k=0.02, alpha=0.05,
                         num_obs_per_period=1440, one_tail=True, upper_tail=True)
        assert len(r['anoms']) == 41


class TestDetectAnomsNegDirection:
    """one_tail=True, upper_tail=False (maps to direction='neg')"""

    def test_anomaly_count(self, raw_data_df):
        r = detect_anoms(raw_data_df.copy(), k=0.02, alpha=0.05,
                         num_obs_per_period=1440, one_tail=True, upper_tail=False)
        assert len(r['anoms']) == 74


class TestDetectAnomsStricterAlpha:
    """alpha=0.01 should find fewer anomalies."""

    def test_fewer_with_stricter_alpha(self, raw_data_df):
        r_strict = detect_anoms(raw_data_df.copy(), k=0.02, alpha=0.01,
                                num_obs_per_period=1440, one_tail=False, upper_tail=True)
        r_default = detect_anoms(raw_data_df.copy(), k=0.02, alpha=0.05,
                                 num_obs_per_period=1440, one_tail=False, upper_tail=True)
        assert len(r_strict['anoms']) <= len(r_default['anoms'])


class TestDetectAnomsInsufficientData:
    """Less than 2 periods of data should return None."""

    def test_returns_none(self, raw_data_df):
        # Only use 1000 rows (< 2 * 1440 = 2880)
        small = raw_data_df.head(1000).copy()
        r = detect_anoms(small, k=0.02, alpha=0.05,
                         num_obs_per_period=1440, one_tail=False, upper_tail=True)
        assert r is None


class TestDetectAnomsNoPeriod:
    """num_obs_per_period=None should raise ValueError."""

    def test_raises(self, raw_data_df):
        with pytest.raises(ValueError, match="must supply period"):
            detect_anoms(raw_data_df.copy(), k=0.02, alpha=0.05,
                         num_obs_per_period=None)
