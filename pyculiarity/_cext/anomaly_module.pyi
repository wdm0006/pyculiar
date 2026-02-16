def median(values: list[float]) -> float:
    """Compute the median of a list of floats."""
    ...

def mad(values: list[float]) -> float:
    """Compute the Median Absolute Deviation (with consistency constant 1.4826)."""
    ...

def t_ppf(p: float, df: int) -> float:
    """Compute the percent-point function (inverse CDF) of Student's t-distribution."""
    ...

def seasonal_decompose(
    values: list[float], period: int
) -> tuple[list[float], list[float], list[float]]:
    """Additive classical seasonal decomposition. Returns (trend, seasonal, remainder)."""
    ...

def esd_test(
    values: list[float],
    max_outliers: int,
    alpha: float,
    one_tail: bool,
    upper_tail: bool,
) -> list[int]:
    """Generalized ESD test. Returns list of 0-based anomaly indices."""
    ...
