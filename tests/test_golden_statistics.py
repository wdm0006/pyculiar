"""Golden tests for statistical primitives.

Locks in exact behavior of median, MAD, and t_ppf against scipy/statsmodels
so the C rewrite can be validated.
"""

import numpy as np
import pytest

from pyculiar._cext.anomaly_module import median as _c_median, mad as _c_mad, t_ppf as _c_t_ppf


# ---------------------------------------------------------------------------
# Median tests
# ---------------------------------------------------------------------------

class TestMedian:
    def test_single_element(self):
        assert _c_median([42.0]) == 42.0

    def test_two_elements(self):
        assert _c_median([1.0, 3.0]) == 2.0

    def test_odd_count(self):
        assert _c_median([1.0, 2.0, 3.0, 4.0, 5.0]) == 3.0

    def test_even_count(self):
        assert _c_median([1.0, 2.0, 3.0, 4.0]) == 2.5

    def test_sorted(self):
        assert _c_median([10.0, 20.0, 30.0]) == 20.0

    def test_reverse_sorted(self):
        assert _c_median([30.0, 20.0, 10.0]) == 20.0

    def test_all_same(self):
        assert _c_median([7.0, 7.0, 7.0, 7.0]) == 7.0

    def test_negatives(self):
        assert _c_median([-5.0, -1.0, 0.0, 1.0, 5.0]) == 0.0

    def test_random_seed42(self):
        np.random.seed(42)
        arr = np.random.randn(100).tolist()
        assert abs(_c_median(arr) - (-0.1269562917797126)) < 1e-10

    def test_random_seed123(self):
        np.random.seed(123)
        arr = np.random.randn(50).tolist()
        expected = float(np.median(arr))
        assert abs(_c_median(arr) - expected) < 1e-10

    def test_large_array(self):
        np.random.seed(99)
        arr = np.random.randn(10000).tolist()
        expected = float(np.median(arr))
        assert abs(_c_median(arr) - expected) < 1e-10


# ---------------------------------------------------------------------------
# MAD tests
# ---------------------------------------------------------------------------

class TestMAD:
    def test_1_to_5(self):
        result = _c_mad([1.0, 2.0, 3.0, 4.0, 5.0])
        assert abs(result - 1.482602218505602) < 1e-10

    def test_all_ones(self):
        assert _c_mad([1.0, 1.0, 1.0, 1.0, 1.0]) == 0.0

    def test_two_elements(self):
        result = _c_mad([1.0, 100.0])
        assert abs(result - 73.38880981602729) < 1e-6

    def test_random_seed42(self):
        np.random.seed(42)
        arr = np.random.randn(100).tolist()
        result = _c_mad(arr)
        assert abs(result - 0.7348122241031572) < 1e-10

    def test_single_element(self):
        assert _c_mad([42.0]) == 0.0

    def test_symmetry(self):
        """MAD should be the same for data and data+constant."""
        arr = [1.0, 2.0, 3.0, 4.0, 5.0]
        shifted = [x + 100.0 for x in arr]
        assert abs(_c_mad(arr) - _c_mad(shifted)) < 1e-10

    def test_scaling(self):
        """MAD(k*x) = |k| * MAD(x)."""
        arr = [1.0, 2.0, 3.0, 4.0, 5.0]
        scaled = [x * 3.0 for x in arr]
        assert abs(_c_mad(scaled) - 3.0 * _c_mad(arr)) < 1e-10

    def test_random_seed7(self):
        np.random.seed(7)
        arr = np.random.randn(200).tolist()
        # Golden value captured from statsmodels.robust.scale.mad
        expected = 0.9711474829488103
        assert abs(_c_mad(arr) - expected) < 1e-6


# ---------------------------------------------------------------------------
# t_ppf tests
# ---------------------------------------------------------------------------

class TestTPPF:
    """Student's t inverse CDF (percent-point function)."""

    GOLDEN_VALUES = {
        (0.9, 1): 3.0776835371752544,
        (0.9, 2): 1.8856180831641272,
        (0.9, 3): 1.6377443536962093,
        (0.9, 5): 1.475884048824481,
        (0.9, 10): 1.372183641110336,
        (0.9, 20): 1.3253407069850467,
        (0.9, 50): 1.2987136941948096,
        (0.9, 100): 1.290074761346516,
        (0.9, 500): 1.2832470207103746,
        (0.9, 1000): 1.2823987214609243,
        (0.95, 1): 6.313751514675037,
        (0.95, 2): 2.9199855803537242,
        (0.95, 3): 2.3533634348018233,
        (0.95, 5): 2.015048373333023,
        (0.95, 10): 1.812461122811676,
        (0.95, 20): 1.7247182429207875,
        (0.95, 50): 1.675905025163097,
        (0.95, 100): 1.6602343260853392,
        (0.95, 500): 1.6479068539295108,
        (0.95, 1000): 1.6463788172854639,
        (0.975, 1): 12.706204736174694,
        (0.975, 2): 4.302652729749462,
        (0.975, 3): 3.1824463052837078,
        (0.975, 5): 2.5705818356363146,
        (0.975, 10): 2.228138851986274,
        (0.975, 20): 2.0859634472658644,
        (0.975, 50): 2.0085591121007607,
        (0.975, 100): 1.983971518523552,
        (0.975, 500): 1.9647198374673676,
        (0.975, 1000): 1.9623390808264078,
        (0.99, 1): 31.820515953773935,
        (0.99, 2): 6.9645567342832715,
        (0.99, 3): 4.540702858568132,
        (0.99, 5): 3.364929998907217,
        (0.99, 10): 2.763769458112696,
        (0.99, 20): 2.5279770027415736,
        (0.99, 50): 2.4032719166741714,
        (0.99, 100): 2.3642173662384818,
        (0.99, 500): 2.333828955352305,
        (0.99, 1000): 2.3300826747555123,
        (0.995, 1): 63.656741162871526,
        (0.995, 2): 9.924843200918287,
        (0.995, 3): 5.840909309733355,
        (0.995, 5): 4.032142983555227,
        (0.995, 10): 3.16927267261695,
        (0.995, 20): 2.8453397097861077,
        (0.995, 50): 2.677793270940844,
        (0.995, 100): 2.6258905214380173,
        (0.995, 500): 2.5856978351419415,
        (0.995, 1000): 2.5807546980659506,
        (0.999, 1): 318.30883898555015,
        (0.999, 2): 22.327124770119866,
        (0.999, 3): 10.214531852407383,
        (0.999, 5): 5.893429531356008,
        (0.999, 10): 4.143700494046589,
        (0.999, 20): 3.5518083432033323,
        (0.999, 50): 3.261409055798318,
        (0.999, 100): 3.173739493738782,
        (0.999, 500): 3.106611624321629,
        (0.999, 1000): 3.098402163912923,
    }

    @pytest.mark.parametrize("p,df", list(GOLDEN_VALUES.keys()))
    def test_golden_values(self, p, df):
        expected = self.GOLDEN_VALUES[(p, df)]
        result = _c_t_ppf(p, df)
        # Allow relative tolerance of 1e-6 for the C implementation
        rel_tol = abs(expected) * 1e-6
        assert abs(result - expected) < max(rel_tol, 1e-9), \
            f"t_ppf({p}, {df}): expected {expected}, got {result}"

    def test_symmetry(self):
        """t_ppf(p, df) = -t_ppf(1-p, df) for symmetric t distribution."""
        for df in [1, 5, 10, 100]:
            for p in [0.9, 0.95, 0.99]:
                upper = _c_t_ppf(p, df)
                lower = _c_t_ppf(1.0 - p, df)
                assert abs(upper + lower) < 1e-8, \
                    f"Symmetry failed for df={df}, p={p}"

    def test_median_is_zero(self):
        """t_ppf(0.5, df) should be 0 for all df."""
        for df in [1, 5, 10, 100, 1000]:
            assert abs(_c_t_ppf(0.5, df)) < 1e-10
