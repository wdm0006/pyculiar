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

import copy
import datetime
import math
from collections import namedtuple
from typing import Literal

import numpy as np
import pandas as pd
from pandas import DataFrame, Timestamp

from pyculiar.detect_anoms import detect_anoms

Direction = namedtuple("Direction", ["one_tail", "upper_tail"])


def detect_ts(
    df: DataFrame,
    max_anoms: float = 0.10,
    direction: Literal["pos", "neg", "both"] = "pos",
    alpha: float = 0.05,
    threshold: Literal["med_max", "p95", "p99"] | None = None,
    e_value: bool = False,
    longterm: bool = False,
    piecewise_median_period_weeks: int = 2,
    granularity: Literal["sec", "min", "hr", "day"] = "day",
    verbose: bool = False,
    inplace: bool = True,
) -> dict[str, DataFrame]:
    """Detect anomalies in a seasonal univariate time series using S-H-ESD.

    Args:
        df: Nonempty two-column DataFrame: Unix timestamps (int64 or float64)
            followed by numeric observations. Timestamps are always Unix seconds.
        max_anoms: Maximum anomaly fraction, default 0.10; must be at most 0.49.
            Clamped to at least one observation divided by the input length.
        direction: Anomaly direction: "pos", "neg", or "both". Defaults to "pos".
        alpha: Statistical significance level for accepting anomalies. Defaults
            to 0.05; the usual range is 0.01 to 0.1.
        threshold: Optional minimum observed value based on daily maxima:
            "med_max" (median), "p95", or "p99". Defaults to None.
        e_value: Include expected values at anomaly timestamps. Defaults to False.
        longterm: Process the series in piecewise windows, recommended for series
            longer than a month. Defaults to False.
        piecewise_median_period_weeks: Window size in weeks for longterm processing.
            Must be at least 2. Defaults to 2.
        granularity: Observation spacing: "sec", "min", "hr", or "day".
            Defaults to "day". Millisecond data is not supported because input
            timestamps are Unix seconds.
        verbose: Pass verbosity to the detector and enable available warnings.
            Defaults to False.
        inplace: Mutate the input frame when True (default), including converting
            timestamps to UTC datetimes. Column names are restored on normal
            completion. False processes a deep copy instead.

    Returns:
        A dictionary with an "anoms" DataFrame containing "timestamp" (Unix
        seconds) and "anoms" (observed values). With e_value=True, it also contains
        "expected_value" (trend plus seasonal component). No detections produce
        an empty frame. Insufficient data produces an empty frame with only
        "timestamp" and "anoms", even when e_value=True.

    Raises:
        ValueError: If the input is empty, has an invalid shape or timestamp/value
            types, or an option fails validation.
    """

    if not isinstance(df, DataFrame):
        raise ValueError("data must be a single data frame.")
    else:
        if len(df.columns) != 2 or not df.iloc[:, 1].map(np.isreal).all():
            raise ValueError("""data must be a 2 column data.frame, with the first column being a set of timestamps, and
                                the second coloumn being numeric values.""")

        if df.dtypes.iloc[0].type is not np.float64 and df.dtypes.iloc[0].type is not np.int64:
            raise ValueError("""The input timestamp column must be a float or integer of the unix timestamp, not date
                                time columns, date strings or pd.TimeStamp columns.""")

    if len(df) == 0:
        raise ValueError("data must contain at least one row")

    # Sanity check all input parameters
    if max_anoms > 0.49:
        length = len(df)
        raise ValueError(
            f"max_anoms must be less than 50% of the data points (max_anoms ={round(max_anoms * length, 0):f} data_points ={length})."
        )

    if direction not in ["pos", "neg", "both"]:
        raise ValueError("direction options are: pos | neg | both.")

    if not math.isfinite(alpha):
        raise ValueError("alpha must be a finite number.")

    if not (0.01 <= alpha <= 0.1) and verbose:
        import warnings

        warnings.warn("alpha is the statistical signifigance, and is usually between 0.01 and 0.1")

    if threshold not in [None, "med_max", "p95", "p99"]:
        raise ValueError("threshold options are: None | med_max | p95 | p99")

    if not isinstance(e_value, bool):
        raise ValueError("e_value must be a boolean")

    if not isinstance(longterm, bool):
        raise ValueError("longterm must be a boolean")

    if piecewise_median_period_weeks < 2:
        raise ValueError("piecewise_median_period_weeks must be at greater than 2 weeks")

    # if the data is daily, then we need to bump the period to weekly to get multiple examples
    gran = granularity
    gran_period = {"sec": 3600, "min": 1440, "hr": 24, "day": 7}
    period = gran_period.get(gran)
    if not period:
        raise ValueError(f"granularity {gran!r} is not supported. Supported values: sec, min, hr, day.")

    if not inplace:
        df = copy.deepcopy(df)

    # change the column names in place, rather than copying the entire dataset, but save the headers to replace them.
    orig_header = df.columns.values
    df.rename(columns={df.columns.values[0]: "timestamp", df.columns.values[1]: "value"}, inplace=True)

    # now convert the timestamp column into a proper timestamp
    df["timestamp"] = df["timestamp"].map(lambda x: datetime.datetime.fromtimestamp(x, tz=datetime.timezone.utc))

    num_obs = len(df.value)

    clamp = 1 / float(num_obs)
    max_anoms = max(max_anoms, clamp)

    if longterm:
        if gran == "day":
            num_obs_in_period = period * piecewise_median_period_weeks + 1
            num_days_in_period = 7 * piecewise_median_period_weeks + 1
        else:
            num_obs_in_period = period * 7 * piecewise_median_period_weeks
            num_days_in_period = 7 * piecewise_median_period_weeks

        last_date = df.timestamp.iloc[-1]

        all_data = []

        for j in range(0, len(df.timestamp), num_obs_in_period):
            start_date = df.timestamp.iloc[j]
            end_date = min(start_date + datetime.timedelta(days=num_obs_in_period), df.timestamp.iloc[-1])

            # if there is at least 14 days left, subset it, otherwise subset last_date - 14days
            if (end_date - start_date).days == num_days_in_period:
                sub_df = df[(df.timestamp >= start_date) & (df.timestamp < end_date)]
            else:
                sub_df = df[
                    (df.timestamp > (last_date - datetime.timedelta(days=num_days_in_period)))
                    & (df.timestamp <= last_date)
                ]
            all_data.append(sub_df)
    else:
        all_data = [df]

    all_anoms = DataFrame(columns=["timestamp", "value"])
    seasonal_plus_trend = DataFrame(columns=["timestamp", "value"])

    # Detect anomalies on all data (either entire data in one-pass, or in 2 week blocks if longterm=TRUE)
    for i in range(len(all_data)):
        directions = {"pos": Direction(True, True), "neg": Direction(True, False), "both": Direction(False, True)}
        anomaly_direction = directions[direction]

        # detect_anoms actually performs the anomaly detection and returns the result in a list containing the anomalies
        # as well as the decomposed components of the time series for further analysis.

        s_h_esd_timestamps = detect_anoms(
            all_data[i],
            k=max_anoms,
            alpha=alpha,
            num_obs_per_period=period,
            use_decomp=True,
            one_tail=anomaly_direction.one_tail,
            upper_tail=anomaly_direction.upper_tail,
            verbose=verbose,
        )
        if s_h_esd_timestamps is None:
            return {"anoms": DataFrame(columns=["timestamp", "anoms"])}

        # store decomposed comps in local variable and overwrite s_h_esd_timestamps to contain only the anom timestamps
        data_decomp = s_h_esd_timestamps["stl"]
        s_h_esd_timestamps = s_h_esd_timestamps["anoms"]

        # -- Step 3: Use detected anomaly timestamps to extract the actual anomalies (timestamp and value) from the data
        if s_h_esd_timestamps:
            anoms = all_data[i][all_data[i].timestamp.isin(s_h_esd_timestamps)]
        else:
            anoms = DataFrame(columns=["timestamp", "value"])

        # Filter the anomalies using one of the thresholding functions if applicable
        if threshold:
            # Calculate daily max values
            periodic_maxes = df.groupby(df.timestamp.map(Timestamp.date)).aggregate(np.max).value

            # Calculate the threshold set by the user
            thresh = 0.5
            if threshold == "med_max":
                thresh = float(periodic_maxes.median())
            elif threshold == "p95":
                thresh = float(periodic_maxes.quantile(0.95))
            elif threshold == "p99":
                thresh = float(periodic_maxes.quantile(0.99))

            # Remove any anoms below the threshold
            anoms = anoms[anoms.value >= thresh]

        all_anoms = pd.concat([all_anoms, anoms])
        seasonal_plus_trend = pd.concat([seasonal_plus_trend, data_decomp])

    # Cleanup potential duplicates
    all_anoms = all_anoms.drop_duplicates(subset=["timestamp"])
    seasonal_plus_trend = seasonal_plus_trend.drop_duplicates(subset=["timestamp"])

    # Calculate number of anomalies as a percentage
    anom_pct = (len(df.value) / float(num_obs)) * 100

    # name the columns back
    df.rename(columns={"timestamp": orig_header[0], "value": orig_header[1]}, inplace=True)

    if anom_pct == 0:
        return {"anoms": None}  # type: ignore[dict-item]  # unreachable for non-empty input

    all_anoms.index = all_anoms.timestamp

    if e_value:
        d = {
            "timestamp": all_anoms.timestamp,
            "anoms": all_anoms.value,
            "expected_value": seasonal_plus_trend[seasonal_plus_trend.timestamp.isin(all_anoms.timestamp)].value,
        }
    else:
        d = {"timestamp": all_anoms.timestamp, "anoms": all_anoms.value}

    anoms = DataFrame(d, index=d["timestamp"].index)

    # convert timestamps back to unix time
    anoms["timestamp"] = anoms["timestamp"].map(pd.Timestamp.timestamp)

    return {"anoms": anoms}
