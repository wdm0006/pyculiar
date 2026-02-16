# MIT License
#
# Copyright (c) 2025 Will McGinnis
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from itertools import groupby

import numpy as np
import pandas as pd

from pyculiar._cext.anomaly_module import seasonal_decompose as c_seasonal_decompose
from pyculiar._cext.anomaly_module import esd_test as c_esd_test


def detect_anoms(
    data, k=0.49, alpha=0.05, num_obs_per_period=None, use_decomp=True, one_tail=True, upper_tail=True, verbose=False
):
    """
    Detects anomalies in a time series using S-H-ESD.

    Args:
        data: Time series to perform anomaly detection on.
        k: Maximum number of anomalies that S-H-ESD will detect as a percentage of the data.
        alpha: The level of statistical significance with which to accept or reject anomalies.
        num_obs_per_period: Defines the number of observations in a single period, and used during seasonal decomposition.
        use_decomp: Use seasonal decomposition during anomaly detection.
        one_tail: If TRUE only positive or negative going anomalies are detected depending on if upper_tail is TRUE or FALSE.
        upper_tail: If TRUE and one_tail is also TRUE, detect only positive going (right-tailed) anomalies.
                    If FALSE and one_tail is TRUE, only detect negative (left-tailed) anomalies.
        verbose: Additionally printing for debugging.

    Returns:
        A dictionary containing the anomalies (anoms) and decomposition components (stl).
    """

    if num_obs_per_period is None:
        raise ValueError("must supply period length for time series decomposition")

    if list(data.columns.values) != ["timestamp", "value"]:
        data.columns = ["timestamp", "value"]

    num_obs = len(data)

    # Check to make sure we have at least two periods worth of data for
    # anomaly context
    if num_obs < num_obs_per_period * 2:
        print("Anom detection needs at least 2 periods worth of data")
        return None

    # run length encode result of isnull, check for internal nulls
    if (
        len(list(x[0] for x in groupby(pd.isnull(pd.concat([pd.Series([np.nan]), data.value, pd.Series([np.nan])])))))
        > 3
    ):
        raise ValueError(
            "Data contains non-leading NAs. We suggest replacing NAs with interpolated values (see na.approx in Zoo package)."
        )
    else:
        data = data.dropna()

    # -- Step 1: Decompose data. This returns a univariate remainder which will be used for anomaly detection.

    data = data.set_index("timestamp")

    if not pd.api.types.is_integer_dtype(data.index):
        resample_period = {1440: "min", 24: "h", 7: "D"}
        resample_period = resample_period.get(num_obs_per_period)
        if not resample_period:
            raise ValueError("Unsupported resample period: %d" % num_obs_per_period)
        data = data.resample(resample_period).mean().dropna()

    # Use C extension for seasonal decomposition
    values = data.value.tolist()
    trend_list, seasonal_list, remainder_list = c_seasonal_decompose(values, num_obs_per_period)

    decomp = pd.DataFrame(
        {
            "trend": trend_list,
            "seasonal": seasonal_list,
            "remainder": remainder_list,
        },
        index=data.index,
    )

    # Remove the seasonal component, and the median of the data to create the
    # univariate remainder
    residuals = [values[i] - seasonal_list[i] - data.value.median() for i in range(len(values))]

    d = {"timestamp": data.index, "value": residuals}
    data = pd.DataFrame(d)

    p = {
        "timestamp": decomp.index,
        "value": pd.to_numeric((decomp["trend"] + decomp["seasonal"]).truncate(), errors="coerce"),
    }
    data_decomp = pd.DataFrame(p)

    # Maximum number of outliers that S-H-ESD can detect (e.g. 49% of data)
    max_outliers = int(num_obs * k)

    if max_outliers == 0:
        raise ValueError(
            "With longterm=TRUE, AnomalyDetection splits the data into 2 week periods by default. You have %d observations in a period, which is too few. Set a higher piecewise_median_period_weeks."
            % num_obs
        )

    # Use C extension for ESD test
    anom_indices = c_esd_test(data.value.tolist(), max_outliers, alpha, one_tail, upper_tail)

    # Map 0-based indices back to DataFrame timestamp values
    if anom_indices:
        timestamps = data.timestamp.tolist()
        R_idx = [timestamps[i] for i in anom_indices]
    else:
        R_idx = None

    return {"anoms": R_idx, "stl": data_decomp}
