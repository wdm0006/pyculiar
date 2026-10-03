# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-08-08

First release since `0.0.5`. The library has been modernized for current Python
and pandas, and its entire numerical core has been rewritten from scratch as a C
extension — `scipy` and `statsmodels` are no longer dependencies.

The `detect_ts` signature, parameters, and results are unchanged, and its outputs
are pinned to the previous implementation by golden tests. **The import name and
the license both changed**, so read the migration notes below before upgrading.

### Migrating from 0.0.5

- **Import from `pyculiar`, not `pyculiarity`:**

  ```python
  # before
  from pyculiarity import detect_ts

  # after
  from pyculiar import detect_ts
  ```

  The PyPI distribution name is unchanged (`pip install pyculiar`); only the
  module you import from moved, so that it matches the package name.

- **The license changed from GPL-3.0 to MIT.** If your usage depended on the
  copyleft terms, review the new [LICENSE](LICENSE).

- **Python 3.10+ and pandas 2.0+ are now required.** Stay on `0.0.5` if you need
  Python 2 or pandas 1.x.

### Removed

- **`scipy`, `statsmodels`, `patsy`, `pytz`, and `future` are no longer
  dependencies.** Runtime requirements are now just `numpy` and `pandas>=2.0`.
- The `pyculiarity.stl` module (`stl()`, a statsmodels-backed LOESS
  decomposition) is gone. Seasonal decomposition now happens inside the C
  extension; call `pyculiar._cext.anomaly_module.seasonal_decompose(values, period)`
  if you were using it directly.
- Python 2 compatibility shims (`future`, `past.builtins`).

### Changed

- **The computational core is now a C extension**
  (`pyculiar._cext.anomaly_module`), implementing median, MAD, Student's-t
  inverse CDF, additive classical seasonal decomposition, and the generalized
  ESD test. The public `detect_ts` API is unchanged and produces the same
  results; a golden-test suite locks in the previous implementation's exact
  numeric output.
- **The license is MIT** (was GPL-3.0).
- **Import name is `pyculiar`** (was `pyculiarity`).
- Minimum Python is 3.10; supported through 3.13.
- Minimum pandas is 2.0 (tested against 2.x and 3.x).
- Packaging moved from `setup.py` + `requirements.txt` to `pyproject.toml`.

### Added

- Prebuilt wheels on PyPI for Linux (x86_64, aarch64), macOS (x86_64, arm64),
  and Windows across Python 3.10–3.13, so most installs no longer compile
  anything. An sdist is published for platforms without a wheel.
- Type stubs for the C extension (`pyculiar/_cext/anomaly_module.pyi`).
- Golden tests covering the statistical primitives, seasonal decomposition,
  `detect_anoms`, and end-to-end `detect_ts`, plus input-validation tests.
- CI running lint and the test suite on Python 3.10–3.13 across Linux and macOS,
  with CodeQL (Python and C) and `pip-audit` scanning.

### Fixed

- **The library works on modern pandas again.** Calls to APIs that pandas has
  since removed — `Series.iget`, `Int64Index`, `convert_objects`, and
  `DataFrame.append` — were replaced with supported equivalents. On pandas 2.x
  these previously raised `AttributeError`, so `detect_ts` was effectively
  unusable there.
- Timestamp conversion and integer-division bugs in the test suite, plus uses of
  the removed `DataFrame.set_value`, which had masked incorrect behaviour under
  Python 3.

[0.2.0]: https://github.com/wdm0006/pyculiar/releases/tag/v0.2.0
