pyculiar
========

Anomaly detection for time series using S-H-ESD (Seasonal Hybrid Extreme Studentized Deviate), inspired by
Twitter's [AnomalyDetection](https://github.com/twitter/AnomalyDetection) R package.

This library implements the algorithm from scratch in C (as a Python extension module) for performance, with only
`numpy` and `pandas` as runtime dependencies. No scipy or statsmodels required.

Licensed under the MIT License.

History
-------

This project descends from two earlier implementations:

- [twitter/AnomalyDetection](https://github.com/twitter/AnomalyDetection) — the original R package by Twitter (2015)
- [nicolasmiller/pyculiarity](https://github.com/nicolasmiller/pyculiarity) — a Python port by Nicolas Steven Miller (2015), which used rpy2 to call R

This fork (by [wdm0006](https://github.com/wdm0006/pyculiarity)) originally replaced the R dependency with
statsmodels and scipy. As of v0.2.0 the computational core has been rewritten from scratch in C, eliminating all
remaining third-party numerical dependencies beyond numpy and pandas.

Installation
------------

    pip install pyculiar

For development:

    pip install -e ".[dev]"

Usage
-----

The main entry point is `detect_ts`, which expects a two-column Pandas DataFrame with integer Unix timestamps and
numeric values.

```python
from pyculiar import detect_ts
import pandas as pd

data = pd.read_csv('tests/raw_data.csv', usecols=['timestamp', 'count'])
data['timestamp'] = pd.to_datetime(data['timestamp']).astype(int) // 10**9

results = detect_ts(data, max_anoms=0.05, alpha=0.001, direction='both', granularity='min')
print(results['anoms'])
```

Parameters:

- `max_anoms`: Maximum anomalies as a fraction of data (0.0 - 0.49)
- `direction`: `'pos'`, `'neg'`, or `'both'`
- `alpha`: Statistical significance level (typically 0.01 - 0.1)
- `granularity`: `'day'`, `'hr'`, `'min'`, `'sec'`, or `'ms'`
- `threshold`: Optional filtering — `None`, `'med_max'`, `'p95'`, or `'p99'`
- `e_value`: Include expected values in output (bool)
- `longterm`: Enable piecewise processing for series > 1 month (bool)

Run the tests
-------------

    pytest

Copyright and License
---------------------

Copyright 2025 Will McGinnis

Licensed under the MIT License. See [LICENSE](LICENSE) for details.

Original R source Copyright 2015 Twitter, Inc and other contributors.
Python port Copyright 2015 Nicolas Steven Miller.
