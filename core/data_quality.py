"""
Generic data-quality checks.

Each check inspects the dataframe + inferred schema and returns issues
in a uniform shape so the dashboard can list them:

    {
        "severity": "critical" | "warning" | "info",
        "check": "missing_values",
        "column": "Revenue" | None,
        "message": "...",
        "value": 0.42,
    }
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .schema_inference import (
    DatasetSchema,
    format_number,
    humanize,
    name_tokens,
)


SEVERITY_ORDER = {"critical": 0, "warning": 1, "info": 2}

# Generic English words for quantities that cannot be negative.
NON_NEGATIVE_NAME_TOKENS = {
    "age", "days", "day", "hours", "hour", "minutes", "seconds",
    "duration", "count", "counts", "qty", "quantity", "units",
    "price", "amount", "revenue", "cost", "sales", "total", "num",
    "number", "distance", "weight", "height", "length", "size",
    "visits", "orders", "items", "since", "tenure",
}


def run_quality_checks(
    dataframe: pd.DataFrame,
    schema: DatasetSchema,
) -> list[dict[str, Any]]:

    issues: list[dict[str, Any]] = []

    issues += _check_duplicates(dataframe)

    for name, column in schema.columns.items():
        series = dataframe[name]

        issues += _check_missing(name, column.missing_ratio)

        if column.role == "empty":
            issues.append(_issue(
                "warning", "empty_column", name,
                f"{humanize(name)} has no values and can be dropped.",
            ))
            continue

        if column.role == "constant":
            issues.append(_issue(
                "info", "constant_column", name,
                f"{humanize(name)} has a single value and carries "
                f"no information.",
            ))
            continue

        if column.role == "measure":
            issues += _check_impossible_negatives(name, series)
            issues += _check_outliers(name, series)
            issues += _check_skew(name, series)

        if column.role in {"dimension", "binary"} and not column.is_numeric:
            issues += _check_inconsistent_labels(name, series)

        if column.role == "binary":
            issues += _check_imbalance(name, series)

        if column.role == "time":
            issues += _check_dates(name, series)

    issues.sort(
        key=lambda issue: (
            SEVERITY_ORDER[issue["severity"]],
            issue["column"] or "",
        )
    )

    return issues


# ----------------------------------------------------------------------
# Checks
# ----------------------------------------------------------------------

def _check_duplicates(dataframe: pd.DataFrame) -> list[dict[str, Any]]:
    duplicates = int(dataframe.duplicated().sum())

    if duplicates == 0:
        return []

    ratio = duplicates / len(dataframe)

    return [_issue(
        "warning" if ratio > 0.01 else "info",
        "duplicate_rows",
        None,
        f"{duplicates:,} fully duplicated rows ({ratio:.1%}).",
        ratio,
    )]


def _check_missing(name: str, ratio: float) -> list[dict[str, Any]]:
    if ratio <= 0 or ratio >= 1:
        return []

    if ratio >= 0.4:
        severity = "critical"
    elif ratio >= 0.05:
        severity = "warning"
    else:
        severity = "info"

    share = f"{ratio:.1%}" if ratio >= 0.001 else "less than 0.1%"

    return [_issue(
        severity, "missing_values", name,
        f"{humanize(name)} is missing in {share} of rows.",
        ratio,
    )]


def _check_impossible_negatives(
    name: str,
    series: pd.Series,
) -> list[dict[str, Any]]:

    if not set(name_tokens(name)) & NON_NEGATIVE_NAME_TOKENS:
        return []

    values = series.dropna()

    if values.empty:
        return []

    ratio = float((values < 0).mean())

    if ratio == 0:
        return []

    return [_issue(
        "critical" if ratio >= 0.05 else "warning",
        "negative_values",
        name,
        f"{humanize(name)} looks like a non-negative quantity but "
        f"{ratio:.1%} of values are negative (min {format_number(values.min())}). "
        f"Check how it is calculated.",
        ratio,
    )]


def _check_outliers(name: str, series: pd.Series) -> list[dict[str, Any]]:
    values = series.dropna().astype(float)

    if len(values) < 20:
        return []

    # Zero-inflated / mostly-constant columns make IQR fences
    # meaningless; the skew check covers them instead.
    if values.value_counts(normalize=True).iloc[0] > 0.5:
        return []

    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1

    if iqr == 0:
        return []

    lower = q1 - 3 * iqr
    upper = q3 + 3 * iqr

    ratio = float(((values < lower) | (values > upper)).mean())

    if ratio < 0.005:
        return []

    bounds = []
    if (values < lower).any():
        bounds.append(f"below {format_number(lower)}")
    if (values > upper).any():
        bounds.append(f"above {format_number(upper)}")

    return [_issue(
        "warning" if ratio >= 0.02 else "info",
        "outliers",
        name,
        f"{humanize(name)} has {ratio:.1%} extreme values "
        f"({' or '.join(bounds)}).",
        ratio,
    )]


def _check_skew(name: str, series: pd.Series) -> list[dict[str, Any]]:
    values = series.dropna().astype(float)

    if len(values) < 20:
        return []

    skew = float(values.skew())

    if not np.isfinite(skew) or abs(skew) < 2:
        return []

    direction = "right" if skew > 0 else "left"

    return [_issue(
        "info", "skewed", name,
        f"{humanize(name)} is strongly {direction}-skewed "
        f"(skew {skew:.1f}); medians describe it better than means.",
        skew,
    )]


def _check_inconsistent_labels(
    name: str,
    series: pd.Series,
) -> list[dict[str, Any]]:

    values = pd.Series(series.dropna().astype(str).unique())

    if values.empty:
        return []

    normalized = values.str.strip().str.lower().str.replace(
        r"\s+", " ", regex=True,
    )

    collisions = normalized[normalized.duplicated(keep=False)]

    if collisions.empty:
        return []

    examples = values[normalized.isin(collisions)].head(4).tolist()

    return [_issue(
        "warning", "inconsistent_labels", name,
        f"{humanize(name)} has labels that differ only by case or "
        f"spacing, e.g. {', '.join(repr(v) for v in examples)}.",
        len(collisions),
    )]


def _check_imbalance(name: str, series: pd.Series) -> list[dict[str, Any]]:
    shares = series.dropna().value_counts(normalize=True)

    if len(shares) != 2:
        return []

    minority_share = float(shares.min())

    if minority_share >= 0.1:
        return []

    minority = shares.idxmin()

    return [_issue(
        "info", "imbalanced", name,
        f"{humanize(name)} is imbalanced: only {minority_share:.1%} "
        f"of rows are '{minority}'.",
        minority_share,
    )]


def _check_dates(name: str, series: pd.Series) -> list[dict[str, Any]]:
    values = pd.to_datetime(series, errors="coerce").dropna()

    if values.empty:
        return []

    issues = []
    now = pd.Timestamp.now()

    future = float((values > now + pd.Timedelta(days=1)).mean())

    if future > 0:
        issues.append(_issue(
            "warning", "future_dates", name,
            f"{humanize(name)} has {future:.1%} dates in the future.",
            future,
        ))

    ancient = float((values < pd.Timestamp("1900-01-01")).mean())

    if ancient > 0:
        issues.append(_issue(
            "warning", "implausible_dates", name,
            f"{humanize(name)} has {ancient:.1%} dates before 1900.",
            ancient,
        ))

    return issues


def _issue(
    severity: str,
    check: str,
    column: str | None,
    message: str,
    value: Any = None,
) -> dict[str, Any]:
    return {
        "severity": severity,
        "check": check,
        "column": column,
        "message": message,
        "value": value,
    }
