/*
 * MIT License
 *
 * Copyright (c) 2025 Will McGinnis
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ========================================================================
 * Internal helpers
 * ======================================================================== */

static void swap_double(double *a, double *b) {
    double t = *a;
    *a = *b;
    *b = t;
}

/* Quickselect: rearrange arr so that arr[k] is the k-th smallest element.
 * Modifies arr in place. */
static void quickselect(double *arr, int n, int k) {
    int lo = 0, hi = n - 1;
    while (lo < hi) {
        double pivot = arr[k];
        int i = lo, j = hi;
        do {
            while (arr[i] < pivot) i++;
            while (arr[j] > pivot) j--;
            if (i <= j) {
                swap_double(&arr[i], &arr[j]);
                i++;
                j--;
            }
        } while (i <= j);
        if (j < k) lo = i;
        if (k < i) hi = j;
    }
}

/* ========================================================================
 * c_median
 * ======================================================================== */

static double c_median(double *data, int n) {
    double *tmp = (double *)malloc(n * sizeof(double));
    if (!tmp) return 0.0;
    memcpy(tmp, data, n * sizeof(double));

    if (n % 2 == 1) {
        quickselect(tmp, n, n / 2);
        double med = tmp[n / 2];
        free(tmp);
        return med;
    } else {
        quickselect(tmp, n, n / 2 - 1);
        double a = tmp[n / 2 - 1];
        /* After quickselect for k=n/2-1, the element at n/2 is the min of
         * the right partition, so find it. */
        double b = tmp[n / 2];
        for (int i = n / 2 + 1; i < n; i++) {
            if (tmp[i] < b) b = tmp[i];
        }
        free(tmp);
        return (a + b) / 2.0;
    }
}

/* ========================================================================
 * c_mad  (Median Absolute Deviation with consistency constant)
 * ======================================================================== */

/* 1 / Phi^{-1}(3/4) — matches statsmodels default */
#define MAD_CONSTANT 1.4826022185056018

static double c_mad(double *data, int n) {
    double med = c_median(data, n);
    double *abs_dev = (double *)malloc(n * sizeof(double));
    if (!abs_dev) return 0.0;
    for (int i = 0; i < n; i++) {
        abs_dev[i] = fabs(data[i] - med);
    }
    double result = c_median(abs_dev, n) * MAD_CONSTANT;
    free(abs_dev);
    return result;
}

/* ========================================================================
 * c_t_ppf  (Student's t inverse CDF / percent-point function)
 *
 * Strategy:
 *   1. Special cases: df=1 (Cauchy), df=2 (closed form)
 *   2. Normal quantile via Acklam's rational approximation
 *   3. Cornish-Fisher expansion to approximate t quantile
 *   4. Newton-Raphson refinement using regularized incomplete beta
 * ======================================================================== */

/* --- Acklam's rational approximation for the standard normal inverse CDF ---
 * Peter J. Acklam's algorithm.  Accurate to ~1.15e-9 relative error. */
static double normal_ppf(double p) {
    /* Coefficients */
    static const double a[] = {
        -3.969683028665376e+01,  2.209460984245205e+02,
        -2.759285104469687e+02,  1.383577518672690e+02,
        -3.066479806614716e+01,  2.506628277459239e+00
    };
    static const double b[] = {
        -5.447609879822406e+01,  1.615858368580409e+02,
        -1.556989798598866e+02,  6.680131188771972e+01,
        -1.328068155288572e+01
    };
    static const double c[] = {
        -7.784894002430293e-03, -3.223964580411365e-01,
        -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00,  2.938163982698783e+00
    };
    static const double d[] = {
         7.784695709041462e-03,  3.224671290700398e-01,
         2.445134137142996e+00,  3.754408661907416e+00
    };

    static const double p_low  = 0.02425;
    static const double p_high = 1.0 - 0.02425;

    double q, r, x;

    if (p <= 0.0) return -INFINITY;
    if (p >= 1.0) return  INFINITY;
    if (p == 0.5) return  0.0;

    if (p < p_low) {
        /* Rational approximation for lower region */
        q = sqrt(-2.0 * log(p));
        x = (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
            ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0);
    } else if (p <= p_high) {
        /* Rational approximation for central region */
        q = p - 0.5;
        r = q * q;
        x = (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5]) * q /
            (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1.0);
    } else {
        /* Rational approximation for upper region */
        q = sqrt(-2.0 * log(1.0 - p));
        x = -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
             ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0);
    }

    return x;
}

/* --- Log-gamma via Lanczos approximation --- */
static double log_gamma(double x) {
    static const double coeff[] = {
        76.18009172947146,    -86.50532032941677,
        24.01409824083091,     -1.231739572450155,
         0.1208650973866179e-2, -0.5395239384953e-5
    };
    double y = x, tmp = x + 5.5;
    tmp -= (x + 0.5) * log(tmp);
    double ser = 1.000000000190015;
    for (int j = 0; j < 6; j++) {
        ser += coeff[j] / (++y);
    }
    return -tmp + log(2.5066282746310005 * ser / x);
}

/* --- Regularized incomplete beta function I_x(a, b) via continued fraction --- */
static double betacf(double a, double b, double x) {
    int maxiter = 200;
    double eps = 3.0e-12;
    double qab = a + b;
    double qap = a + 1.0;
    double qam = a - 1.0;
    double c2 = 1.0;
    double d2 = 1.0 - qab * x / qap;
    if (fabs(d2) < 1.0e-30) d2 = 1.0e-30;
    d2 = 1.0 / d2;
    double h = d2;

    for (int m = 1; m <= maxiter; m++) {
        int m2 = 2 * m;
        /* Even step */
        double aa = (double)m * (b - (double)m) * x /
                    (((double)qam + (double)m2) * (a + (double)m2));
        d2 = 1.0 + aa * d2;
        if (fabs(d2) < 1.0e-30) d2 = 1.0e-30;
        c2 = 1.0 + aa / c2;
        if (fabs(c2) < 1.0e-30) c2 = 1.0e-30;
        d2 = 1.0 / d2;
        h *= d2 * c2;
        /* Odd step */
        aa = -(a + (double)m) * (qab + (double)m) * x /
              ((a + (double)m2) * (qap + (double)m2));
        d2 = 1.0 + aa * d2;
        if (fabs(d2) < 1.0e-30) d2 = 1.0e-30;
        c2 = 1.0 + aa / c2;
        if (fabs(c2) < 1.0e-30) c2 = 1.0e-30;
        d2 = 1.0 / d2;
        double del = d2 * c2;
        h *= del;
        if (fabs(del - 1.0) < eps) break;
    }
    return h;
}

static double betainc(double a, double b, double x) {
    if (x < 0.0 || x > 1.0) return 0.0;
    if (x == 0.0 || x == 1.0) return x;

    double lbeta = log_gamma(a + b) - log_gamma(a) - log_gamma(b)
                 + a * log(x) + b * log(1.0 - x);

    if (x < (a + 1.0) / (a + b + 2.0)) {
        return exp(lbeta) * betacf(a, b, x) / a;
    } else {
        return 1.0 - exp(lbeta) * betacf(b, a, 1.0 - x) / b;
    }
}

/* --- t-distribution CDF --- */
static double t_cdf(double t_val, int df) {
    double x = (double)df / ((double)df + t_val * t_val);
    double ib = betainc((double)df / 2.0, 0.5, x);
    if (t_val >= 0.0) {
        return 1.0 - 0.5 * ib;
    } else {
        return 0.5 * ib;
    }
}

/* --- t-distribution PDF --- */
static double t_pdf(double t_val, int df) {
    double ddf = (double)df;
    return exp(log_gamma((ddf + 1.0) / 2.0) - log_gamma(ddf / 2.0)
               - 0.5 * log(ddf * M_PI)
               - ((ddf + 1.0) / 2.0) * log(1.0 + t_val * t_val / ddf));
}

static double c_t_ppf(double p, int df) {
    /* Special case: p = 0.5 → 0 for all df */
    if (p == 0.5) return 0.0;

    /* Special case: df = 1 (Cauchy distribution) */
    if (df == 1) {
        return tan(M_PI * (p - 0.5));
    }

    /* Special case: df = 2 — closed form: t = (2p-1) / sqrt(2*p*(1-p)) */
    if (df == 2) {
        return (2.0 * p - 1.0) / sqrt(2.0 * p * (1.0 - p));
    }

    /* General case: use normal approximation + Newton-Raphson refinement */
    double z = normal_ppf(p);

    /* Cornish-Fisher expansion: convert z → t approximation */
    double ddf = (double)df;
    double z2 = z * z;

    /* First-order correction */
    double g1 = (z2 + 1.0) / (4.0 * ddf);
    /* Second-order correction */
    double g2 = (5.0 * z2 * z2 + 16.0 * z2 + 3.0) / (96.0 * ddf * ddf);
    /* Third-order correction */
    double g3 = (3.0 * z2 * z2 * z2 + 19.0 * z2 * z2 + 17.0 * z2 - 15.0) /
                (384.0 * ddf * ddf * ddf);

    double t_approx = z + z * g1 + z * g2 + z * g3;

    /* Newton-Raphson refinement */
    for (int iter = 0; iter < 12; iter++) {
        double cdf_val = t_cdf(t_approx, df);
        double pdf_val = t_pdf(t_approx, df);
        if (pdf_val < 1e-300) break;
        double delta = (cdf_val - p) / pdf_val;
        t_approx -= delta;
        if (fabs(delta) < 1e-14 * (1.0 + fabs(t_approx))) break;
    }

    return t_approx;
}

/* ========================================================================
 * c_seasonal_decompose
 *
 * Additive classical decomposition (matches statsmodels seasonal_decompose):
 *   1. Centered moving average for trend
 *   2. Detrend: detrended = data - trend
 *   3. Seasonal: period-wise average of detrended, then center
 *   4. Remainder: data - trend - seasonal
 *   5. Fill NaN at edges with 0.0
 * ======================================================================== */

/* Computes trend, seasonal, remainder in caller-provided arrays.
 * All arrays must be pre-allocated with n elements.
 * NaN edges in trend are set to 0.0. */
static void c_seasonal_decompose(const double *data, int n, int period,
                                  double *trend, double *seasonal,
                                  double *remainder) {
    int i, j;

    /* --- Step 1: Centered moving average for trend --- */
    /* For even period: 2xperiod convolution (first average of period,
     * then average of 2 consecutive period-averages).
     * This matches statsmodels' behavior for even periods. */

    /* Initialize trend to 0 (will be NaN-equivalent at edges) */
    for (i = 0; i < n; i++) trend[i] = 0.0;

    if (period % 2 == 0) {
        /* Even period: 2x moving average
         * First pass: simple MA of length 'period' */
        int half = period / 2;
        double *ma1 = (double *)calloc(n, sizeof(double));
        if (!ma1) return;

        /* ma1[i] = mean(data[i..i+period-1]) for i in [0, n-period] */
        double sum = 0.0;
        for (i = 0; i < period; i++) sum += data[i];
        ma1[0] = sum / (double)period;
        for (i = 1; i <= n - period; i++) {
            sum += data[i + period - 1] - data[i - 1];
            ma1[i] = sum / (double)period;
        }

        /* Second pass: average consecutive ma1 values
         * trend[half .. n-half-1] = (ma1[i-half] + ma1[i-half+1]) / 2
         * But we need to align properly:
         * ma1[i] corresponds to center at i + (period-1)/2.0
         * For even period, center is at i + period/2 - 0.5
         * So trend[k] = (ma1[k-half] + ma1[k-half+1]) / 2 when valid
         *
         * Actually, statsmodels does:
         *   filt = np.array([1/(2*period)] + [1/period]*(period-1) + [1/(2*period)])
         *   convolved with data, then trim
         * Which is equivalent to:
         *   trend[i] = (0.5*data[i-half] + data[i-half+1] + ... + data[i+half-1] + 0.5*data[i+half]) / period
         */
        for (i = half; i < n - half; i++) {
            double s = 0.5 * data[i - half];
            for (j = i - half + 1; j < i + half; j++) {
                s += data[j];
            }
            s += 0.5 * data[i + half];
            trend[i] = s / (double)period;
        }

        free(ma1);
    } else {
        /* Odd period: simple centered MA */
        int half = period / 2;
        for (i = half; i < n - half; i++) {
            double s = 0.0;
            for (j = i - half; j <= i + half; j++) {
                s += data[j];
            }
            trend[i] = s / (double)period;
        }
    }

    /* Mark edges: trend is 0 at edges (NaN filled with 0) */
    /* Already initialized to 0, so nothing more needed */

    /* --- Step 2: Detrended values --- */
    double *detrended = (double *)malloc(n * sizeof(double));
    if (!detrended) return;
    for (i = 0; i < n; i++) {
        if (trend[i] != 0.0) {
            detrended[i] = data[i] - trend[i];
        } else {
            detrended[i] = data[i]; /* Edge: treat as detrended = data */
        }
    }

    /* --- Step 3: Seasonal component --- */
    /* Average detrended values at each position within the period */
    double *period_avg = (double *)calloc(period, sizeof(double));
    int *period_count = (int *)calloc(period, sizeof(int));
    if (!period_avg || !period_count) {
        free(detrended);
        free(period_avg);
        free(period_count);
        return;
    }

    /* Only use values where trend is valid (non-edge) */
    int half_p = (period % 2 == 0) ? period / 2 : period / 2;
    for (i = 0; i < n; i++) {
        int pos = i % period;
        if (i >= half_p && i < n - half_p) {
            period_avg[pos] += detrended[i];
            period_count[pos]++;
        }
    }
    for (i = 0; i < period; i++) {
        if (period_count[i] > 0) {
            period_avg[i] /= (double)period_count[i];
        }
    }

    /* Center the seasonal component (subtract its mean) */
    double smean = 0.0;
    for (i = 0; i < period; i++) smean += period_avg[i];
    smean /= (double)period;
    for (i = 0; i < period; i++) period_avg[i] -= smean;

    /* Tile seasonal across the full length */
    for (i = 0; i < n; i++) {
        seasonal[i] = period_avg[i % period];
    }

    /* --- Step 4: Remainder --- */
    for (i = 0; i < n; i++) {
        remainder[i] = data[i] - trend[i] - seasonal[i];
    }

    free(detrended);
    free(period_avg);
    free(period_count);
}

/* ========================================================================
 * c_esd_test  (Generalized ESD test for anomaly detection)
 *
 * data: residuals (data - seasonal - median)
 * n: length
 * k: maximum fraction of anomalies
 * alpha: significance level
 * one_tail: 1 for one-tailed test
 * upper_tail: 1 for upper-tail (only used if one_tail=1)
 * out_indices: pre-allocated array of at least max_outliers ints
 * Returns: number of anomalies detected
 * ======================================================================== */

static int c_esd_test(const double *data, int n, int max_outliers,
                       double alpha, int one_tail, int upper_tail,
                       int *out_indices) {
    /* Working copy of data and index tracking */
    double *work = (double *)malloc(n * sizeof(double));
    int *orig_idx = (int *)malloc(n * sizeof(int));
    if (!work || !orig_idx) {
        free(work);
        free(orig_idx);
        return 0;
    }
    memcpy(work, data, n * sizeof(double));
    for (int i = 0; i < n; i++) orig_idx[i] = i;

    int *R_idx = (int *)malloc(max_outliers * sizeof(int));
    if (!R_idx) { free(work); free(orig_idx); return 0; }

    int num_anoms = 0;
    int cur_n = n;

    for (int i = 1; i <= max_outliers; i++) {
        /* Compute median of current working set */
        double med = c_median(work, cur_n);

        /* Compute MAD */
        double data_sigma = c_mad(work, cur_n);
        if (data_sigma == 0.0) break;

        /* Find the observation with maximum test statistic */
        double max_val = -1.0;
        int max_j = 0;
        for (int j = 0; j < cur_n; j++) {
            double ares;
            if (one_tail) {
                if (upper_tail) {
                    ares = (work[j] - med) / data_sigma;
                } else {
                    ares = (med - work[j]) / data_sigma;
                }
            } else {
                ares = fabs(work[j] - med) / data_sigma;
            }
            if (ares > max_val) {
                max_val = ares;
                max_j = j;
            }
        }

        double R = max_val;
        R_idx[i - 1] = orig_idx[max_j];

        /* Remove the outlier by shifting */
        for (int j = max_j; j < cur_n - 1; j++) {
            work[j] = work[j + 1];
            orig_idx[j] = orig_idx[j + 1];
        }
        cur_n--;

        /* Compute critical value */
        double p;
        if (one_tail) {
            p = 1.0 - alpha / (double)(n - i + 1);
        } else {
            p = 1.0 - alpha / (double)(2 * (n - i + 1));
        }

        double t = c_t_ppf(p, (n - i - 1));
        double lam = t * (double)(n - i) /
                     sqrt(((double)(n - i - 1) + t * t) * (double)(n - i + 1));

        if (R > lam) {
            num_anoms = i;
        }
    }

    /* Copy results */
    if (num_anoms > 0) {
        memcpy(out_indices, R_idx, num_anoms * sizeof(int));
    }

    free(work);
    free(orig_idx);
    free(R_idx);
    return num_anoms;
}


/* ========================================================================
 * Python wrapper functions
 * ======================================================================== */

/* Helper: parse a Python list of floats into a C double array.
 * Returns NULL on failure (sets Python exception). Caller must free(). */
static double *list_to_doubles(PyObject *list, int *out_n) {
    if (!PyList_Check(list)) {
        PyErr_SetString(PyExc_TypeError, "expected a list of floats");
        return NULL;
    }
    Py_ssize_t n = PyList_GET_SIZE(list);
    if (n <= 0) {
        PyErr_SetString(PyExc_ValueError, "list must not be empty");
        return NULL;
    }
    double *arr = (double *)malloc(n * sizeof(double));
    if (!arr) {
        PyErr_NoMemory();
        return NULL;
    }
    for (Py_ssize_t i = 0; i < n; i++) {
        PyObject *item = PyList_GET_ITEM(list, i);
        arr[i] = PyFloat_AsDouble(item);
        if (arr[i] == -1.0 && PyErr_Occurred()) {
            free(arr);
            return NULL;
        }
    }
    *out_n = (int)n;
    return arr;
}

/* --- median(values: list[float]) -> float --- */
static PyObject *py_median(PyObject *self, PyObject *args) {
    PyObject *list;
    if (!PyArg_ParseTuple(args, "O", &list)) return NULL;

    int n;
    double *data = list_to_doubles(list, &n);
    if (!data) return NULL;

    double result = c_median(data, n);
    free(data);
    return PyFloat_FromDouble(result);
}

/* --- mad(values: list[float]) -> float --- */
static PyObject *py_mad(PyObject *self, PyObject *args) {
    PyObject *list;
    if (!PyArg_ParseTuple(args, "O", &list)) return NULL;

    int n;
    double *data = list_to_doubles(list, &n);
    if (!data) return NULL;

    double result = c_mad(data, n);
    free(data);
    return PyFloat_FromDouble(result);
}

/* --- t_ppf(p: float, df: int) -> float --- */
static PyObject *py_t_ppf(PyObject *self, PyObject *args) {
    double p;
    int df;
    if (!PyArg_ParseTuple(args, "di", &p, &df)) return NULL;

    if (df < 1) {
        PyErr_SetString(PyExc_ValueError, "df must be >= 1");
        return NULL;
    }

    double result = c_t_ppf(p, df);
    return PyFloat_FromDouble(result);
}

/* --- seasonal_decompose(values: list[float], period: int)
 *     -> tuple[list[float], list[float], list[float]] --- */
static PyObject *py_seasonal_decompose(PyObject *self, PyObject *args) {
    PyObject *list;
    int period;
    if (!PyArg_ParseTuple(args, "Oi", &list, &period)) return NULL;

    int n;
    double *data = list_to_doubles(list, &n);
    if (!data) return NULL;

    if (period < 2) {
        free(data);
        PyErr_SetString(PyExc_ValueError, "period must be >= 2");
        return NULL;
    }

    double *trend = (double *)calloc(n, sizeof(double));
    double *seasonal = (double *)calloc(n, sizeof(double));
    double *remainder = (double *)calloc(n, sizeof(double));
    if (!trend || !seasonal || !remainder) {
        free(data); free(trend); free(seasonal); free(remainder);
        return PyErr_NoMemory();
    }

    c_seasonal_decompose(data, n, period, trend, seasonal, remainder);

    /* Build Python lists */
    PyObject *py_trend = PyList_New(n);
    PyObject *py_seasonal = PyList_New(n);
    PyObject *py_remainder = PyList_New(n);
    if (!py_trend || !py_seasonal || !py_remainder) {
        Py_XDECREF(py_trend);
        Py_XDECREF(py_seasonal);
        Py_XDECREF(py_remainder);
        free(data); free(trend); free(seasonal); free(remainder);
        return NULL;
    }

    for (int i = 0; i < n; i++) {
        PyList_SET_ITEM(py_trend, i, PyFloat_FromDouble(trend[i]));
        PyList_SET_ITEM(py_seasonal, i, PyFloat_FromDouble(seasonal[i]));
        PyList_SET_ITEM(py_remainder, i, PyFloat_FromDouble(remainder[i]));
    }

    free(data); free(trend); free(seasonal); free(remainder);

    PyObject *result = PyTuple_Pack(3, py_trend, py_seasonal, py_remainder);
    Py_DECREF(py_trend);
    Py_DECREF(py_seasonal);
    Py_DECREF(py_remainder);
    return result;
}

/* --- esd_test(values: list[float], max_outliers: int, alpha: float,
 *              one_tail: bool, upper_tail: bool) -> list[int] --- */
static PyObject *py_esd_test(PyObject *self, PyObject *args) {
    PyObject *list;
    int max_outliers;
    double alpha;
    int one_tail, upper_tail;
    if (!PyArg_ParseTuple(args, "Oidpp", &list, &max_outliers, &alpha,
                          &one_tail, &upper_tail))
        return NULL;

    int n;
    double *data = list_to_doubles(list, &n);
    if (!data) return NULL;

    if (max_outliers <= 0) {
        free(data);
        PyErr_SetString(PyExc_ValueError, "max_outliers must be > 0");
        return NULL;
    }

    int *indices = (int *)malloc(max_outliers * sizeof(int));
    if (!indices) { free(data); return PyErr_NoMemory(); }

    int num_anoms = c_esd_test(data, n, max_outliers, alpha,
                                one_tail, upper_tail, indices);

    PyObject *result = PyList_New(num_anoms);
    if (!result) { free(data); free(indices); return NULL; }

    for (int i = 0; i < num_anoms; i++) {
        PyList_SET_ITEM(result, i, PyLong_FromLong(indices[i]));
    }

    free(data);
    free(indices);
    return result;
}


/* ========================================================================
 * Module definition
 * ======================================================================== */

static PyMethodDef module_methods[] = {
    {"median", py_median, METH_VARARGS,
     "median(values: list[float]) -> float\n\nCompute the median of a list of floats."},
    {"mad", py_mad, METH_VARARGS,
     "mad(values: list[float]) -> float\n\nCompute the Median Absolute Deviation (with consistency constant 1.4826)."},
    {"t_ppf", py_t_ppf, METH_VARARGS,
     "t_ppf(p: float, df: int) -> float\n\nCompute the percent-point function (inverse CDF) of Student's t-distribution."},
    {"seasonal_decompose", py_seasonal_decompose, METH_VARARGS,
     "seasonal_decompose(values: list[float], period: int) -> tuple[list[float], list[float], list[float]]\n\n"
     "Additive classical seasonal decomposition. Returns (trend, seasonal, remainder)."},
    {"esd_test", py_esd_test, METH_VARARGS,
     "esd_test(values: list[float], max_outliers: int, alpha: float, one_tail: bool, upper_tail: bool) -> list[int]\n\n"
     "Generalized ESD test. Returns list of 0-based anomaly indices."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef anomaly_module = {
    PyModuleDef_HEAD_INIT,
    "anomaly_module",
    "C extension for anomaly detection statistical primitives.",
    -1,
    module_methods
};

PyMODINIT_FUNC PyInit_anomaly_module(void) {
    return PyModule_Create(&anomaly_module);
}
