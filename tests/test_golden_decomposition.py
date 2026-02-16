"""Golden tests for seasonal decomposition.

Locks in exact seasonal_decompose behavior using C extension.
"""

import datetime
import os

import numpy as np
import pandas as pd
import pytest

from pyculiarity._cext.anomaly_module import seasonal_decompose


@pytest.fixture
def raw_values():
    """Load raw_data.csv and return as a list of floats."""
    path = os.path.dirname(os.path.realpath(__file__))
    data = pd.read_csv(
        os.path.join(path, 'raw_data.csv'),
        usecols=['timestamp', 'count'])
    return data['count'].tolist()


@pytest.fixture
def decomposed(raw_values):
    """Decompose raw_values with period=1440."""
    trend, seasonal, remainder = seasonal_decompose(raw_values, 1440)
    return np.array(trend), np.array(seasonal), np.array(remainder)


class TestDecompositionShape:
    def test_output_length(self, raw_values):
        trend, seasonal, remainder = seasonal_decompose(raw_values, 1440)
        assert len(trend) == 14398
        assert len(seasonal) == 14398
        assert len(remainder) == 14398


class TestDecompositionNoNaN:
    """All values should be finite (no NaN)."""

    def test_no_nan_in_trend(self, decomposed):
        trend, _, _ = decomposed
        assert not np.any(np.isnan(trend))

    def test_no_nan_in_seasonal(self, decomposed):
        _, seasonal, _ = decomposed
        assert not np.any(np.isnan(seasonal))

    def test_no_nan_in_remainder(self, decomposed):
        _, _, remainder = decomposed
        assert not np.any(np.isnan(remainder))


class TestDecompositionEdges:
    """Trend edges (first/last half-period) should be 0."""

    def test_trend_leading_zeros(self, decomposed):
        trend, _, _ = decomposed
        half_period = 1440 // 2
        assert np.all(trend[:half_period] == 0)

    def test_trend_trailing_zeros(self, decomposed):
        trend, _, _ = decomposed
        half_period = 1440 // 2
        assert np.all(trend[-half_period:] == 0)


class TestSeasonalPeriodicity:
    """Seasonal component should repeat exactly with the given period."""

    def test_first_two_periods_match(self, decomposed):
        _, seasonal, _ = decomposed
        np.testing.assert_array_equal(seasonal[:1440], seasonal[1440:2880])

    def test_second_third_periods_match(self, decomposed):
        _, seasonal, _ = decomposed
        np.testing.assert_array_equal(seasonal[1440:2880], seasonal[2880:4320])


class TestDecompositionStatistics:
    """Summary statistics as checksums for the decomposition."""

    def test_trend_mean(self, decomposed):
        trend, _, _ = decomposed
        assert abs(np.mean(trend) - 101.74061854958877) < 1e-6

    def test_seasonal_mean(self, decomposed):
        _, seasonal, _ = decomposed
        assert abs(np.mean(seasonal) - (-0.006204518068635639)) < 1e-6

    def test_seasonal_std(self, decomposed):
        _, seasonal, _ = decomposed
        assert abs(np.std(seasonal, ddof=1) - 22.27605278018129) < 1e-4

    def test_trend_std(self, decomposed):
        trend, _, _ = decomposed
        assert abs(np.std(trend, ddof=1) - 35.25989824655103) < 1e-4


class TestAdditiveReconstruction:
    """trend + seasonal + remainder should reconstruct the original (where trend is non-zero)."""

    def test_reconstruction_in_middle(self, raw_values, decomposed):
        trend, seasonal, remainder = decomposed
        original = np.array(raw_values)
        reconstructed = trend + seasonal + remainder
        # Only check the middle section where trend is non-zero
        half_period = 1440 // 2
        mid = slice(half_period, -half_period)
        np.testing.assert_allclose(
            reconstructed[mid], original[mid], atol=1e-10)
