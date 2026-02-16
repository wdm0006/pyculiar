import os

import pandas as pd
import pytest

from pyculiarity import detect_ts


@pytest.fixture
def midnight_data():
    path = os.path.dirname(os.path.realpath(__file__))
    data = pd.read_csv(
        os.path.join(path, 'midnight_test_data.csv'),
        usecols=['date', 'value'])
    data['date'] = pd.to_datetime(data['date']).map(pd.Timestamp.timestamp).astype(int)
    return data


def test_check_midnight_date_format(midnight_data):
    results = detect_ts(midnight_data, max_anoms=0.2, threshold=None,
                        direction='both', e_value=True, granularity='hr')
    assert len(results['anoms'].anoms) == len(results['anoms'].expected_value)
