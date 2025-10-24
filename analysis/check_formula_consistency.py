"""Utilities for validating column formulas within a :class:`pandas.DataFrame`."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


_BACKTICK_PATTERN = re.compile(r"`([^`]+)`")


@dataclass
class _EvaluationPlan:
    """Container describing how a formula will be evaluated."""

    normalized_expr: str
    referenced_columns: Sequence[str]


def _detect_referenced_columns(expr: str) -> List[str]:
    """Return the columns explicitly referenced via backticks in *expr*."""

    return _BACKTICK_PATTERN.findall(expr)


def _ensure_backticks(expr: str, candidates: Iterable[str]) -> _EvaluationPlan:
    """Ensure DataFrame columns with special characters are backticked."""

    columns_in_expr: List[str] = []
    normalized = expr

    for column in sorted(set(candidates), key=len, reverse=True):
        # Skip columns that already appear backticked to avoid double quoting
        pattern = rf"(?<!`)({re.escape(column)})(?!`)"

        def _wrap(match: re.Match[str]) -> str:
            columns_in_expr.append(column)
            return f"`{match.group(1)}`"

        normalized, replaced = re.subn(pattern, _wrap, normalized)
        if replaced:
            columns_in_expr.append(column)

    # Include any columns explicitly backticked by the caller
    for column in _detect_referenced_columns(normalized):
        columns_in_expr.append(column)

    # Preserve the order in which the columns were encountered
    seen = set()
    ordered_columns = []
    for column in columns_in_expr:
        if column not in seen:
            ordered_columns.append(column)
            seen.add(column)

    return _EvaluationPlan(normalized_expr=normalized, referenced_columns=ordered_columns)


def _relative_difference(expected: pd.Series, diff: pd.Series) -> pd.Series:
    """Compute a relative difference Series resilient to zeros."""

    denominator = expected.abs()
    rel = pd.Series(np.zeros(len(expected)), index=expected.index, dtype=float)

    non_zero = denominator > 0
    rel.loc[non_zero] = diff.loc[non_zero] / denominator.loc[non_zero]

    zero_expected = ~non_zero
    if zero_expected.any():
        matches_zero = diff.loc[zero_expected] == 0
        rel.loc[zero_expected & matches_zero] = 0.0
        rel.loc[zero_expected & ~matches_zero] = np.nan

    return rel


def check_formula_consistency(
    df: pd.DataFrame,
    formulas: Dict[str, str],
    atol: float = 0.1,
    rtol: float = 0.0,
    show_cols: Optional[List[str]] = None,
    verbose: bool = True,
    return_df: bool = False,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    """Compare a set of formulas against the provided DataFrame.

    Parameters
    ----------
    df
        The DataFrame containing the raw data.
    formulas
        Mapping of ``label -> "target = expression"`` formulas to evaluate.
    atol, rtol
        Absolute and relative tolerances forwarded to :func:`numpy.isclose`.
    show_cols
        Additional columns included in the per-row diagnostics.
    verbose
        When ``True`` print a human readable summary for each formula.
    return_df
        When ``True`` return both the summary and the per-row inconsistencies.

    Returns
    -------
    pandas.DataFrame or tuple[pandas.DataFrame, pandas.DataFrame]
        Summary dataframe (and optionally the per-row inconsistencies).

    Examples
    --------
    >>> formulas = {
    ...     "EUI_method1": "SiteEUI(kBtu/sf) = SiteEnergyUse(kBtu) / PropertyGFATotal",
    ... }
    >>> summary, details = check_formula_consistency(
    ...     df,
    ...     formulas=formulas,
    ...     show_cols=["PropertyName"],
    ...     return_df=True,
    ... )
    """

    if not isinstance(formulas, dict):
        raise TypeError("'formulas' must be a mapping of {label: 'target = expression'}.")

    results: List[Dict[str, object]] = []
    inconsistencies: List[pd.DataFrame] = []

    show_cols = [col for col in (show_cols or []) if col in df.columns]

    for label, equation in formulas.items():
        if "=" not in equation:
            raise ValueError(f"Formula '{label}' is missing '=' (expected 'target = expression').")

        target, raw_expr = equation.split("=", 1)
        target = target.strip().strip("`")
        expr = raw_expr.strip()

        if target not in df.columns:
            raise KeyError(f"Target column '{target}' is not present in the DataFrame (formula '{label}').")

        evaluation_plan = _ensure_backticks(expr, df.columns)

        try:
            computed = df.eval(evaluation_plan.normalized_expr, engine="python", local_dict={"np": np})
        except Exception as exc:  # pragma: no cover - passthrough for debugging clarity
            raise KeyError(f"Failed to evaluate formula '{label}': {exc}") from exc

        mask_valid = (
            df[target].notna()
            & computed.notna()
            & np.isfinite(df[target])
            & np.isfinite(computed)
        )

        comparison = pd.Series(np.isclose(df[target], computed, atol=atol, rtol=rtol), index=df.index)
        comparison &= mask_valid

        diff = (df[target] - computed).abs()

        valid_diff = diff[mask_valid]
        valid_expected = computed[mask_valid]

        mean_diff = float(valid_diff.mean()) if not valid_diff.empty else np.nan
        mean_rel_diff = float(_relative_difference(valid_expected, valid_diff).mean()) if not valid_diff.empty else np.nan
        max_abs_diff = float(valid_diff.max()) if not valid_diff.empty else np.nan
        consistency = float(comparison[mask_valid].mean() * 100) if mask_valid.any() else np.nan
        n_valid = int(mask_valid.sum())
        n_fail = int((~comparison & mask_valid).sum())

        results.append(
            {
                "formula_name": label,
                "target": target,
                "formula": evaluation_plan.normalized_expr,
                "max_abs_diff": max_abs_diff,
                "mean_diff": mean_diff,
                "mean_rel_diff_%": mean_rel_diff * 100 if not np.isnan(mean_rel_diff) else np.nan,
                "consistency_%": consistency,
            }
        )

        if verbose:
            icon = "✅" if consistency == 100 else ("⚠️" if consistency and consistency >= 50 else "❌")
            mean_rel_display = mean_rel_diff * 100 if not np.isnan(mean_rel_diff) else np.nan
            print(f"\n{icon} [{label}]")
            print(f"│   ├─ Cible : {target}")
            print(f"│   ├─ Cohérence : {consistency:6.2f}%\t| Diff. relative : {mean_rel_display:6.2f}%")
            print(f"│   ├─ Diff. moyenne : {mean_diff:8.4f}\t| Max diff : {max_abs_diff:10.4f}")
            print(f"│   └─ Lignes comparées : {n_valid:5d}\t| Incohérentes : {n_fail:5d}")

        mask_inconsistent = mask_valid & ~comparison
        if mask_inconsistent.any():
            referenced_cols = [col for col in evaluation_plan.referenced_columns if col in df.columns]

            context_cols = list(dict.fromkeys(show_cols + [target] + referenced_cols))
            context_cols = [col for col in context_cols if col in df.columns]

            df_bad = df.loc[mask_inconsistent, context_cols].copy()
            df_bad.insert(0, "formula_name", label)
            df_bad.insert(1, "row_index", df_bad.index)
            df_bad["target"] = target
            df_bad["actual"] = df.loc[mask_inconsistent, target]
            df_bad["expected"] = computed.loc[mask_inconsistent]
            df_bad["difference"] = diff.loc[mask_inconsistent]
            df_bad["formula_fields"] = [referenced_cols] * len(df_bad)

            for column in df_bad.select_dtypes(include=[np.floating]).columns:
                df_bad[column] = df_bad[column].round(4)

            base_cols = [
                "row_index",
                "formula_name",
                "target",
                "actual",
                "expected",
                "difference",
                "formula_fields",
            ]

            other_cols = [col for col in df_bad.columns if col not in base_cols]
            ordered_cols = base_cols + other_cols
            df_bad = df_bad.loc[:, [col for col in ordered_cols if col in df_bad.columns]]

            inconsistencies.append(df_bad)

    df_results = pd.DataFrame(results, columns=[
        "formula_name",
        "target",
        "formula",
        "max_abs_diff",
        "mean_diff",
        "mean_rel_diff_%",
        "consistency_%",
    ])

    if inconsistencies:
        df_inconsistencies = pd.concat(inconsistencies, ignore_index=True)
    else:
        df_inconsistencies = pd.DataFrame(
            columns=["formula_name", "row_index", "target", "actual", "expected", "difference", "formula_fields"] + show_cols
        )

    if return_df:
        return df_results, df_inconsistencies

    return df_results.set_index("formula_name")["consistency_%"].to_dict()
