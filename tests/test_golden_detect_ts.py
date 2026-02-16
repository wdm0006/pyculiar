"""Golden tests for detect_ts() end-to-end behavior.

Captures exact anomaly detection results from the scipy/statsmodels
implementation so the C rewrite can be validated.
"""

import copy
import os

import pandas as pd
import pytest

from pyculiarity import detect_ts


@pytest.fixture
def raw_data():
    path = os.path.dirname(os.path.realpath(__file__))
    data = pd.read_csv(
        os.path.join(path, 'raw_data.csv'),
        usecols=['timestamp', 'count'])
    data['timestamp'] = pd.to_datetime(data['timestamp']).map(pd.Timestamp.timestamp).astype(int)
    return data


@pytest.fixture
def midnight_data():
    path = os.path.dirname(os.path.realpath(__file__))
    data = pd.read_csv(
        os.path.join(path, 'midnight_test_data.csv'),
        usecols=['date', 'value'])
    data['date'] = pd.to_datetime(data['date']).map(pd.Timestamp.timestamp).astype(int)
    return data


# ---------------------------------------------------------------------------
# raw_data.csv with granularity='min'
# ---------------------------------------------------------------------------

class TestRawDataBothDirections:
    """direction='both', max_anoms=0.02"""

    EXPECTED_COUNT = 114
    EXPECTED_FIRST_TS = 338745900.0
    EXPECTED_LAST_TS = 339601080.0

    def test_anomaly_count(self, raw_data):
        results = detect_ts(raw_data, max_anoms=0.02, direction='both', granularity='min')
        assert len(results['anoms']) == self.EXPECTED_COUNT

    def test_columns(self, raw_data):
        results = detect_ts(raw_data, max_anoms=0.02, direction='both', granularity='min')
        assert list(results['anoms'].columns) == ['timestamp', 'anoms']

    def test_first_last_timestamp(self, raw_data):
        results = detect_ts(raw_data, max_anoms=0.02, direction='both', granularity='min')
        anoms = results['anoms']
        assert anoms.iloc[0].timestamp == self.EXPECTED_FIRST_TS
        assert anoms.iloc[-1].timestamp == self.EXPECTED_LAST_TS

    def test_exact_timestamps(self, raw_data):
        results = detect_ts(raw_data, max_anoms=0.02, direction='both', granularity='min')
        ts = results['anoms'].timestamp.tolist()
        # Verify count and boundary values
        assert len(ts) == self.EXPECTED_COUNT
        assert ts[0] == self.EXPECTED_FIRST_TS
        assert ts[-1] == self.EXPECTED_LAST_TS


class TestRawDataPosDirection:
    """direction='pos', max_anoms=0.02"""

    EXPECTED_COUNT = 41

    def test_anomaly_count(self, raw_data):
        results = detect_ts(raw_data, max_anoms=0.02, direction='pos', granularity='min')
        assert len(results['anoms']) == self.EXPECTED_COUNT

    def test_first_timestamp(self, raw_data):
        results = detect_ts(raw_data, max_anoms=0.02, direction='pos', granularity='min')
        assert results['anoms'].iloc[0].timestamp == 339057600.0


class TestRawDataNegDirection:
    """direction='neg', max_anoms=0.02"""

    EXPECTED_COUNT = 74

    def test_anomaly_count(self, raw_data):
        results = detect_ts(raw_data, max_anoms=0.02, direction='neg', granularity='min')
        assert len(results['anoms']) == self.EXPECTED_COUNT

    def test_first_timestamp(self, raw_data):
        results = detect_ts(raw_data, max_anoms=0.02, direction='neg', granularity='min')
        assert results['anoms'].iloc[0].timestamp == 338745900.0


class TestRawDataLongterm:
    """direction='both', longterm=True, e_value=True"""

    EXPECTED_COUNT = 114

    def test_anomaly_count(self, raw_data):
        results = detect_ts(raw_data, max_anoms=0.02, direction='both',
                            longterm=True, e_value=True, granularity='min')
        assert len(results['anoms']) == self.EXPECTED_COUNT

    def test_three_columns(self, raw_data):
        results = detect_ts(raw_data, max_anoms=0.02, direction='both',
                            longterm=True, e_value=True, granularity='min')
        assert len(results['anoms'].columns) == 3
        assert list(results['anoms'].columns) == ['timestamp', 'anoms', 'expected_value']


class TestRawDataThreshold:
    """direction='both', threshold='med_max', e_value=True"""

    EXPECTED_COUNT = 4

    def test_anomaly_count(self, raw_data):
        results = detect_ts(raw_data, max_anoms=0.02, direction='both',
                            threshold='med_max', e_value=True, granularity='min')
        assert len(results['anoms']) == self.EXPECTED_COUNT


class TestRawDataMaxAnoms05:
    """direction='both', max_anoms=0.05"""

    EXPECTED_COUNT = 114

    def test_anomaly_count(self, raw_data):
        results = detect_ts(raw_data, max_anoms=0.05, direction='both', granularity='min')
        assert len(results['anoms']) == self.EXPECTED_COUNT


class TestRawDataStricterAlpha:
    """direction='both', alpha=0.01 — stricter significance yields fewer anomalies"""

    EXPECTED_COUNT = 95

    def test_anomaly_count(self, raw_data):
        results = detect_ts(raw_data, max_anoms=0.02, direction='both',
                            alpha=0.01, granularity='min')
        assert len(results['anoms']) == self.EXPECTED_COUNT

    def test_fewer_than_default(self, raw_data):
        """Stricter alpha should find fewer or equal anomalies than default alpha=0.05."""
        r_strict = detect_ts(copy.deepcopy(raw_data), max_anoms=0.02, direction='both',
                             alpha=0.01, granularity='min')
        r_default = detect_ts(copy.deepcopy(raw_data), max_anoms=0.02, direction='both',
                              alpha=0.05, granularity='min')
        assert len(r_strict['anoms']) <= len(r_default['anoms'])


# ---------------------------------------------------------------------------
# midnight_test_data.csv with granularity='hr'
# ---------------------------------------------------------------------------

class TestMidnightData:
    """Hourly midnight data, direction='both', e_value=True"""

    EXPECTED_COUNT = 7
    EXPECTED_TIMESTAMPS = [
        1424998800.0, 1425340800.0, 1425423600.0,
        1425506400.0, 1425682800.0, 1426132800.0, 1427673600.0
    ]

    def test_anomaly_count(self, midnight_data):
        results = detect_ts(midnight_data, max_anoms=0.2, threshold=None,
                            direction='both', e_value=True, granularity='hr')
        assert len(results['anoms']) == self.EXPECTED_COUNT

    def test_anoms_equals_expected_value_length(self, midnight_data):
        results = detect_ts(midnight_data, max_anoms=0.2, threshold=None,
                            direction='both', e_value=True, granularity='hr')
        assert len(results['anoms'].anoms) == len(results['anoms'].expected_value)

    def test_exact_timestamps(self, midnight_data):
        results = detect_ts(midnight_data, max_anoms=0.2, threshold=None,
                            direction='both', e_value=True, granularity='hr')
        assert results['anoms'].timestamp.tolist() == self.EXPECTED_TIMESTAMPS
