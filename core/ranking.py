# def score_pattern(pattern):
#     """
#     Calculate a heuristic interestingness score
#     for a detected pattern.
#     """

#     pattern_type = pattern.get("pattern_type")

#     score = 0

#     # -----------------------------
#     # Correlation patterns
#     # -----------------------------
#     if pattern_type in [
#         "strong_correlation",
#         "moderate_correlation",
#         "weak_correlation"
#     ]:
#         strength = abs(
#             pattern.get("correlation", 0)
#         )

#         score += strength * 100

#     # -----------------------------
#     # Group difference patterns
#     # -----------------------------
#     elif pattern_type == "group_difference":

#         spread_ratio = pattern.get(
#             "spread_ratio"
#         )

#         if spread_ratio is not None:

#             # Larger difference = more interesting
#             if spread_ratio >= 3:
#                 score += 100

#             elif spread_ratio >= 2:
#                 score += 75

#             elif spread_ratio >= 1.5:
#                 score += 50

#             else:
#                 score += 25

#     # -----------------------------
#     # Time trend patterns
#     # -----------------------------
#     elif pattern_type == "time_trend":

#         change_pct = abs(
#             pattern.get("change_pct", 0)
#         )

#         # Larger change = more interesting
#         if change_pct >= 50:
#             score += 100

#         elif change_pct >= 25:
#             score += 75

#         elif change_pct >= 10:
#             score += 50

#         else:
#             score += 20

#     return round(score, 2)


import math


def score_pattern(pattern):
    """
    Calculate a heuristic interestingness score
    for a detected pattern.
    """

    pattern_type = pattern.get("pattern_type")

    score = 0

    # -----------------------------------------
    # Correlation
    # -----------------------------------------
    if pattern_type in [
        "strong_correlation",
        "moderate_correlation",
        "weak_correlation"
    ]:

        correlation = abs(
            pattern.get("correlation", 0)
        )

        score = correlation * 100

    # -----------------------------------------
    # Group difference
    # -----------------------------------------
    elif pattern_type == "group_difference":

        highest = float(
            pattern.get("highest_value", 0)
        )

        lowest = float(
            pattern.get("lowest_value", 0)
        )

        difference = abs(
            highest - lowest
        )

        scale = max(
            abs(highest),
            abs(lowest)
        )

        if scale == 0:
            score = 0

        else:
            relative_difference = (
                difference / scale
            )

            # Convert to a bounded score.
            # This prevents most differences
            # from automatically becoming 100.
            score = (
                relative_difference /
                (1 + relative_difference)
            ) * 100

    # -----------------------------------------
    # Time trend
    # -----------------------------------------
    elif pattern_type == "time_trend":

        r2 = pattern.get(
            "r2",
            0
        )

        normalized_slope = pattern.get(
            "normalized_slope",
            0
        )

        # Convert slope magnitude into
        # percentage-like movement
        trend_strength = (
            normalized_slope * 100
        )

        # Cap movement contribution
        trend_strength = min(
            trend_strength,
            100
        )

        # Combine magnitude and consistency
        score = (
            trend_strength * 0.6
            +
            r2 * 100 * 0.4
        )

    return round(score, 2)



def rank_patterns(patterns, top_n=10):
    """
    Rank detected patterns by interestingness.
    """

    ranked_patterns = []

    for pattern in patterns:

        score = score_pattern(pattern)

        ranked_pattern = pattern.copy()

        ranked_pattern["interestingness_score"] = score

        ranked_patterns.append(
            ranked_pattern
        )

    ranked_patterns.sort(
        key=lambda x: x["interestingness_score"],
        reverse=True
    )

    return ranked_patterns[:top_n]

    