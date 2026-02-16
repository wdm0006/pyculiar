import os

import pandas as pd
import pytest

from pyculiar import detect_ts


@pytest.fixture
def raw_data():
    path = os.path.dirname(os.path.realpath(__file__))
    data = pd.read_csv(
        os.path.join(path, 'raw_data.csv'),
        usecols=['timestamp', 'count'])
    data['timestamp'] = pd.to_datetime(data['timestamp']).map(pd.Timestamp.timestamp).astype(int)
    return data


def test_both_directions(raw_data):
    results = detect_ts(raw_data, max_anoms=0.02, direction='both', granularity='min')
    assert len(results['anoms'].columns) == 2
    assert len(results['anoms'].iloc[:, 1]) > 0


def test_both_directions_e_value_longterm(raw_data):
    results = detect_ts(raw_data, max_anoms=0.02, direction='both',
                        longterm=True, e_value=True, granularity='min')
    assert len(results['anoms'].columns) == 3
    assert len(results['anoms'].iloc[:, 1]) > 0


def test_both_directions_e_value_threshold_med_max(raw_data):
    results = detect_ts(raw_data, max_anoms=0.02, direction='both',
                        threshold="med_max", e_value=True, granularity='min')
    assert len(results['anoms'].columns) == 3
    assert len(results['anoms'].iloc[:, 1]) > 0
