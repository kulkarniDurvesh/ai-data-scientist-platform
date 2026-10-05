
import numpy as np
import pandas as pd
import re


RATE_SUFFIXES = (
    "rate",
    "percentage",
    "percent",
    "ratio",
)


CORRELATION_REDUNDANCY_THRESHOLD = 0.85


def normalize_column_name(column):
    return re.sub(
        r"[^a-z0-9]",
        "",
        str(column).lower()
    )


def is_rate_metric(column):
    normalized = normalize_column_name(column)

    return normalized.endswith(
        RATE_SUFFIXES
    )


def get_measure_columns(pattern):
    if pattern.get("pattern_type") not in [
        "strong_correlation",
        "moderate_correlation",
        "weak_correlation"
    ]:
        return None, None

    return (
        pattern.get("x"),
        pattern.get("y")
    )


def get_canonical_pair(x, y):
    return tuple(
        sorted([
            str(x),
            str(y)
        ])
    )


def remove_duplicate_relationships(patterns):
    """
    Remove exact duplicate relationships.

    A-B and B-A are treated as the same relationship.
    """

    unique_patterns = []
    seen_pairs = set()

    for pattern in patterns:

        x, y = get_measure_columns(
            pattern
        )

        if x is None or y is None:
            unique_patterns.append(
                pattern
            )
            continue

        pair = get_canonical_pair(
            x,
            y
        )

        if pair in seen_pairs:
            continue

        seen_pairs.add(pair)

        unique_patterns.append(
            pattern
        )

    return unique_patterns


def calculate_ratio_error(
    numerator,
    denominator,
    rate
):
    """
    Check whether:

        rate ≈ numerator / denominator

    Returns relative error.
    """

    data = pd.DataFrame({
        "numerator": numerator,
        "denominator": denominator,
        "rate": rate
    }).dropna()

    data = data[
        data["denominator"] != 0
    ]

    if len(data) < 10:
        return None

    expected_rate = (
        data["numerator"] /
        data["denominator"]
    )

    actual_rate = data["rate"]

    error = np.mean(
        np.abs(
            actual_rate -
            expected_rate
        )
    )

    scale = np.mean(
        np.abs(actual_rate)
    )

    if scale == 0:
        return None

    return error / scale


def is_mathematically_derived(
    df,
    x,
    y
):
    """
    Determine whether one metric appears
    to be mathematically derived from another.

    Currently tests:

        rate ≈ numerator / denominator
    """

    if (
        x not in df.columns
        or
        y not in df.columns
    ):
        return False

    if (
        is_rate_metric(x)
        and
        not is_rate_metric(y)
    ):
        rate_column = x
        base_column = y

    elif (
        is_rate_metric(y)
        and
        not is_rate_metric(x)
    ):
        rate_column = y
        base_column = x

    else:
        return False

    numeric_columns = (
        df
        .select_dtypes(
            include=np.number
        )
        .columns
    )

    for denominator in numeric_columns:

        if denominator in [
            rate_column,
            base_column
        ]:
            continue

        error = calculate_ratio_error(
            df[base_column],
            df[denominator],
            df[rate_column]
        )

        if error is None:
            continue

        if error <= 0.05:
            return True

    return False


def is_derived_metric_relationship(
    pattern,
    df
):
    x, y = get_measure_columns(
        pattern
    )

    if x is None or y is None:
        return False

    return is_mathematically_derived(
        df,
        x,
        y
    )


def are_redundant_correlations(
    pattern_a,
    pattern_b,
    threshold=CORRELATION_REDUNDANCY_THRESHOLD
):
    """
    Determine whether two correlation patterns
    represent substantially overlapping relationships.

    Example:

        A ↔ B
        A ↔ C

    If B ↔ C is also highly correlated, these
    relationships may represent the same underlying
    analytical story.

    The check is completely column-name independent.
    """

    x1, y1 = get_measure_columns(
        pattern_a
    )

    x2, y2 = get_measure_columns(
        pattern_b
    )

    if (
        x1 is None
        or y1 is None
        or x2 is None
        or y2 is None
    ):
        return False

    pair_a = {
        str(x1),
        str(y1)
    }

    pair_b = {
        str(x2),
        str(y2)
    }

    # Exact same relationship.
    if pair_a == pair_b:
        return True

    correlation_a = abs(
        pattern_a.get(
            "correlation",
            0
        )
    )

    correlation_b = abs(
        pattern_b.get(
            "correlation",
            0
        )
    )

    # We only consider very strong relationships
    # candidates for redundancy.
    if (
        correlation_a < threshold
        or
        correlation_b < threshold
    ):
        return False

    # If the relationships share a variable,
    # they may describe the same underlying pattern.
    shared_columns = (
        pair_a &
        pair_b
    )

    if not shared_columns:
        return False

    return True


def remove_redundant_correlations(
    patterns,
    threshold=CORRELATION_REDUNDANCY_THRESHOLD
):
    """
    Reduce repeated high-correlation stories.

    Important:
    This function does NOT modify the detected
    patterns themselves. It only reduces the set
    used for final insight presentation.

    The strongest correlation is retained first.
    """

    correlation_patterns = []
    other_patterns = []

    for pattern in patterns:

        x, y = get_measure_columns(
            pattern
        )

        if x is None or y is None:
            other_patterns.append(
                pattern
            )
            continue

        correlation_patterns.append(
            pattern
        )

    correlation_patterns.sort(
        key=lambda pattern: abs(
            pattern.get(
                "correlation",
                0
            )
        ),
        reverse=True
    )

    selected = []

    for pattern in correlation_patterns:

        redundant = False

        for selected_pattern in selected:

            if are_redundant_correlations(
                pattern,
                selected_pattern,
                threshold
            ):
                redundant = True
                break

        if not redundant:
            selected.append(
                pattern
            )

    return (
        selected +
        other_patterns
    )


def filter_redundant_patterns(
    patterns,
    df
):
    """
    Complete insight filtering pipeline.

    Steps:

    1. Remove mathematically derived metrics.
    2. Remove exact duplicate relationships.
    3. Reduce highly redundant correlations.
    """

    filtered_patterns = []

    for pattern in patterns:

        if is_derived_metric_relationship(
            pattern,
            df
        ):
            continue

        filtered_patterns.append(
            pattern
        )

    filtered_patterns = (
        remove_duplicate_relationships(
            filtered_patterns
        )
    )

    filtered_patterns = (
        remove_redundant_correlations(
            filtered_patterns
        )
    )

    return filtered_patterns

