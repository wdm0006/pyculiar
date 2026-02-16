import os

import pandas as pd
import numpy as np
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


def test_handling_of_leading_trailing_nas(raw_data):
    for i in list(range(10)) + [len(raw_data) - 1]:
        raw_data.at[i, 'count'] = np.nan

    results = detect_ts(raw_data, max_anoms=0.02, direction='both', granularity='min')
    assert len(results['anoms'].columns) == 2
    assert len(results['anoms'].iloc[:, 1]) > 0


def test_handling_of_middle_nas(raw_data):
    raw_data.at[len(raw_data) // 2, 'count'] = np.nan
    with pytest.raises(ValueError):
        detect_ts(raw_data, max_anoms=0.02, direction='both', granularity='min')
