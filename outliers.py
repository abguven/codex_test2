"""Utility helpers for identifying outliers in pandas series."""

from __future__ import annotations

from numbers import Real
from typing import Iterable, Sequence, Tuple

import pandas as pd


def get_outlier_bounds(
    series: pd.Series,
    compute_method: str | None = None,
    q: Iterable[float] | None = None,
    verbose: bool = False,
    iqr_coefficient: float = 1.5,
    *,
    coef: float | None = None,
) -> Tuple[float, float]:
    """Return the lower and upper bounds for detecting outliers.

    Parameters
    ----------
    series:
        Series whose bounds are being calculated. Missing values are ignored.
    compute_method:
        Strategy for computing the bounds. ``"iqr"`` uses the interquartile range
        and ``"quantile"`` expects explicit quantile bounds via ``q``. When left as
        ``None`` the method defaults to ``"quantile"`` if ``q`` is provided,
        otherwise ``"iqr"`` is used.
    q:
        Iterable containing the lower and upper quantile bounds used when
        ``compute_method`` is ``"quantile"``. At least two values must be provided.
    verbose:
        When ``True``, print diagnostic information with icons describing the
        computation steps.
    iqr_coefficient:
        Multiplier applied to the IQR when ``compute_method`` is ``"iqr"``.
    coef:
        Deprecated alias for ``iqr_coefficient`` maintained for backwards
        compatibility.

    Returns
    -------
    Tuple[float, float]
        ``(lower, upper)`` bounds beyond which values are considered outliers.
    """

    cleaned = series.dropna()
    if cleaned.empty:
        _emit(verbose, "result", "No data available after dropping missing values.")
        return float("nan"), float("nan")

    if coef is not None:
        iqr_coefficient = coef

    if compute_method is None:
        compute_method = "quantile" if q is not None else "iqr"

    if q is not None and compute_method != "quantile":
        raise ValueError(
            "q parameter requires compute_method to be 'quantile' or None"
        )

    if compute_method == "iqr":
        if iqr_coefficient <= 0:
            raise ValueError("iqr_coefficient must be positive")

        q1, q3 = cleaned.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower = q1 - iqr_coefficient * iqr
        upper = q3 + iqr_coefficient * iqr

        _emit(verbose, "quartiles", f"Q1={q1} | Q3={q3} | IQR={iqr}")
        _emit(
            verbose,
            "bounds",
            (
                "Using IQR bounds: "
                f"coef={iqr_coefficient}, lower={lower}, upper={upper}"
            ),
        )
    elif compute_method == "quantile":
        if q is None:
            raise ValueError("q parameter is required when compute_method='quantile'")

        lower_quantile, upper_quantile = _normalise_quantiles(q)
        quantile_bounds = cleaned.quantile([lower_quantile, upper_quantile])
        lower, upper = tuple(quantile_bounds.tolist())

        _emit(
            verbose,
            "bounds",
            (
                "Using quantile bounds: "
                f"q_low={lower_quantile} -> {lower}, "
                f"q_high={upper_quantile} -> {upper}"
            ),
        )
    else:
        raise ValueError("Unsupported compute_method: {0}".format(compute_method))

    mask = (cleaned < lower) | (cleaned > upper)
    outliers = int(mask.sum())
    pct_outliers = (outliers / len(cleaned)) * 100 if len(cleaned) else 0.0
    _emit(
        verbose,
        "outliers",
        f"Outliers: {outliers}/{len(cleaned)} ({pct_outliers:.1f}%)",
    )

    if verbose and outliers > 0:
        examples = cleaned[mask].head(5).tolist()

        def _format_example(value: object) -> str:
            if isinstance(value, Real):
                return f"{float(value):.1f}"
            return str(value)

        formatted = ", ".join(_format_example(value) for value in examples)
        remaining = outliers - len(examples)
        suffix = f" (+ {remaining} autres)" if remaining > 0 else ""
        _emit(verbose, "examples", f"Exemples: [{formatted}]{suffix}")

    return lower, upper


def _normalise_quantiles(q: Iterable[float]) -> Tuple[float, float]:
    """Normalise the provided iterable of quantiles into a ``(low, high)`` tuple."""

    if isinstance(q, Sequence):
        quantiles = list(q)
    else:
        quantiles = list(q)  # type: ignore[arg-type]

    if len(quantiles) < 2:
        raise ValueError("q parameter should contain at least 2 elements")

    low, high = quantiles[0], quantiles[1]
    if not (0 <= low <= 1 and 0 <= high <= 1):
        raise ValueError("quantile values must be between 0 and 1")

    if low > high:
        raise ValueError("quantile lower bound must not exceed upper bound")

    return float(low), float(high)


_VERBOSE_ICONS = {
    "info": "ℹ️",
    "quartiles": "📊",
    "bounds": "🎯",
    "outliers": "⚠️",
    "examples": "🔍",
    "result": "✅",
}


def _emit(verbose: bool, icon_key: str, message: str) -> None:
    """Emit a verbose message prefixed with an icon when enabled."""

    if not verbose:
        return

    icon = _VERBOSE_ICONS.get(icon_key, _VERBOSE_ICONS["info"])
    print(f"{icon} {message}")


def compute_outlier_count(
    series: pd.Series,
    compute_method: str | None = None,
    q: Iterable[float] | None = None,
    verbose: bool = False,
    iqr_coefficient: float = 1.5,
) -> int:
    """Count the number of outliers in ``series``.

    Parameters
    ----------
    series:
        Values to inspect for outliers. Missing values are ignored.
    compute_method:
        Strategy for computing the bounds. ``"iqr"`` uses the interquartile range
        and ``"quantile"`` expects explicit quantile bounds via ``q``. When left as
        ``None`` the method defaults to ``"quantile"`` if ``q`` is provided,
        otherwise ``"iqr"`` is used.
    q:
        Iterable containing the lower and upper quantile bounds used when
        ``compute_method`` is ``"quantile"``. At least two values must be provided.

    verbose:
        When ``True``, print diagnostic information with icons describing the
        computation steps.
    iqr_coefficient:
        Multiplier applied to the IQR when ``compute_method`` is ``"iqr"``.

    Returns
    -------
    int
        Number of values considered outliers.
    """

    if compute_method is None:
        compute_method = "quantile" if q is not None else "iqr"

    if q is not None and compute_method != "quantile":
        raise ValueError(
            "q parameter requires compute_method to be 'quantile' or None"
        )

    cleaned = series.dropna()
    dropped = len(series) - len(cleaned)
    _emit(verbose, "info", f"Removed {dropped} missing value(s) before analysis.")
    if cleaned.empty:
        _emit(verbose, "result", "No data available after dropping missing values.")
        return 0

    lower, upper = get_outlier_bounds(
        cleaned,
        compute_method=compute_method,
        q=q,
        verbose=verbose,
        iqr_coefficient=iqr_coefficient,
    )

    mask = (cleaned < lower) | (cleaned > upper)
    outliers = int(mask.sum())
    _emit(verbose, "result", f"Identified {outliers} outlier(s).")
    return outliers


def compute_iqr_malus(
    series: pd.Series,
    compute_method: str | None = None,
    q: Iterable[float] | None = None,
    verbose: bool = False,
    iqr_coefficient: float = 1.5,
) -> pd.Series:
    """Return a binary malus flag marking outliers according to the bounds.

    The function mirrors :func:`compute_outlier_count` by supporting both IQR and
    quantile-based bounds while returning a 0/1 Series aligned with ``series``.
    """

    if compute_method is None:
        compute_method = "quantile" if q is not None else "iqr"

    if q is not None and compute_method != "quantile":
        raise ValueError(
            "q parameter requires compute_method to be 'quantile' or None"
        )

    cleaned = series.dropna()
    dropped = len(series) - len(cleaned)
    _emit(verbose, "info", f"Removed {dropped} missing value(s) before analysis.")
    if cleaned.empty:
        _emit(verbose, "result", "No data available after dropping missing values.")
        return pd.Series(0, index=series.index, dtype=int)

    lower, upper = get_outlier_bounds(
        cleaned,
        compute_method=compute_method,
        q=q,
        verbose=verbose,
        iqr_coefficient=iqr_coefficient,
    )

    mask = (series < lower) | (series > upper)
    outliers = int(mask.fillna(False).sum())
    pct_outliers = mask.mean(skipna=True) * 100 if len(series) else 0.0
    _emit(
        verbose,
        "result",
        (
            f"Generated malus for {outliers} outlier(s) "
            f"({pct_outliers:.1f}% of observations)."
        ),
    )

    return mask.fillna(False).astype(int)

