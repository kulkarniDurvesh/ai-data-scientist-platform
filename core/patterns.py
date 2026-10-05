import pandas as pd
import numpy as np

def detect_measure_relationship_pattern(
    candidate,
    statistics
):
    """
    Detect patterns between two numerical measures.
    """

    x = candidate["x"]
    y = candidate["y"]

    pearson = statistics.get("pearson")
    spearman = statistics.get("spearman")

    if pearson is None:
        return None

    abs_pearson = abs(pearson)
    abs_spearman = abs(spearman) if spearman is not None else 0

    # Use the stronger evidence from the two correlations
    strength = max(
        abs_pearson,
        abs_spearman
    )

    if strength >= 0.70:

        if pearson > 0:
            direction = "positive"
        else:
            direction = "negative"

        return {
            "pattern_type": "strong_correlation",
            "x": x,
            "y": y,
            "direction": direction,
            "correlation": pearson,
            "strength": strength,
            "description": (
                f"Strong {direction} association detected "
                f"between {x} and {y}."
            )
        }

    elif strength >= 0.50:

        if pearson > 0:
            direction = "positive"
        else:
            direction = "negative"

        return {
            "pattern_type": "moderate_correlation",
            "x": x,
            "y": y,
            "direction": direction,
            "correlation": pearson,
            "strength": strength,
            "description": (
                f"Moderate {direction} association detected "
                f"between {x} and {y}."
            )
        }

    elif strength >= 0.30:

        if pearson > 0:
            direction = "positive"
        else:
            direction = "negative"

        return {
            "pattern_type": "weak_correlation",
            "x": x,
            "y": y,
            "direction": direction,
            "correlation": pearson,
            "strength": strength,
            "description": (
                f"Weak-to-moderate {direction} association "
                f"detected between {x} and {y}."
            )
        }

    return None

def detect_dimension_measure_pattern(
    candidate,
    statistics
):
    """
    Detect patterns between a categorical dimension
    and numerical measure.
    """

    dimension = candidate["dimension"]
    measure = candidate["measure"]

    if statistics is None or statistics.empty:
        return None

    mean_column = "mean"

    if mean_column not in statistics.columns:
        return None

    # Remove groups where mean is missing
    data = statistics.dropna(
        subset=[mean_column]
    )

    if data.empty:
        return None

    highest_row = data.loc[
        data[mean_column].idxmax()
    ]

    lowest_row = data.loc[
        data[mean_column].idxmin()
    ]

    highest_value = highest_row[mean_column]
    lowest_value = lowest_row[mean_column]

    if lowest_value == 0:
        spread_ratio = None
    else:
        spread_ratio = (
            highest_value / lowest_value
        )

    return {
        "pattern_type": "group_difference",
        "dimension": dimension,
        "measure": measure,
        "highest_group": highest_row[dimension],
        "highest_value": highest_value,
        "lowest_group": lowest_row[dimension],
        "lowest_value": lowest_value,
        "spread_ratio": spread_ratio,
        "description": (
            f"{highest_row[dimension]} has the highest "
            f"average {measure}, while "
            f"{lowest_row[dimension]} has the lowest "
            f"average {measure}."
        )
    }

def detect_time_trend_pattern(
    candidate,
    statistics
):
    """
    Detect an overall trend in a time-based analysis
    using linear regression slope and R².
    """

    time_column = candidate["time"]
    measure = candidate["measure"]

    if statistics is None or statistics.empty:
        return None

    if "mean" not in statistics.columns:
        return None

    data = statistics.dropna(
        subset=["mean"]
    ).copy()

    if len(data) < 3:
        return None

    # Get time-series values
    values = data["mean"].astype(float).values

    # Calculate trend strength
    slope, r2, normalized_slope = calculate_trend_strength(
        values
    )

    # -----------------------------------------
    # Calculate trend change percentage
    # -----------------------------------------
    mean_value = np.mean(np.abs(values))

    if mean_value == 0:
        change_pct = 0
    else:
        # Estimated total change across the period
        total_change = slope * (len(values) - 1)

        change_pct = (
            total_change / mean_value
        ) * 100

    # -----------------------------------------
    # Determine trend
    # -----------------------------------------

    # Require both:
    # 1. reasonable trend consistency
    # 2. meaningful movement
    if r2 < 0.50 or normalized_slope < 0.01:

        trend = "stable"

    elif slope > 0:

        trend = "increasing"

    else:

        trend = "decreasing"

    return {
        "pattern_type": "time_trend",
        "time": time_column,
        "measure": measure,
        "trend": trend,

        # New trend metrics
        "change_pct": round(change_pct, 2),
        "r2": round(r2, 4),
        "normalized_slope": round(
            normalized_slope,
            4
        ),

        "description": (
            f"{measure} shows an overall "
            f"{trend} trend over time."
        )
    }

def detect_pattern(
    candidate,
    statistics
):

    candidate_type = candidate["type"]

    if candidate_type == "measure_measure":

        return detect_measure_relationship_pattern(
            candidate,
            statistics
        )

    elif candidate_type == "dimension_measure":

        return detect_dimension_measure_pattern(
            candidate,
            statistics
        )

    elif candidate_type == "time_measure":

        return detect_time_trend_pattern(
            candidate,
            statistics
        )

    return None

def calculate_trend_strength(values):

    y = np.asarray(
        values,
        dtype=float
    )

    if len(y) < 3:
        return 0, 0, 0

    x = np.arange(len(y))

    slope, intercept = np.polyfit(
        x,
        y,
        1
    )

    # Avoid correlation warning
    # when all values are identical
    if np.std(y) == 0:

        correlation = 0

    else:

        correlation = np.corrcoef(
            x,
            y
        )[0, 1]

        if np.isnan(correlation):
            correlation = 0

    r2 = correlation ** 2

    mean_value = np.mean(
        np.abs(y)
    )

    if mean_value == 0:

        normalized_slope = 0

    else:

        normalized_slope = (
            abs(slope) /
            mean_value
        )

    return (
        slope,
        r2,
        normalized_slope
    )
