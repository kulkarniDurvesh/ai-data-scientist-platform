
def explain_correlation(pattern):
    """
    Generate a human-readable explanation
    for a correlation pattern.
    """

    x = pattern.get("x")
    y = pattern.get("y")

    direction = pattern.get(
        "direction",
        "unknown"
    )

    strength = pattern.get(
        "strength",
        0
    )

    correlation = pattern.get(
        "correlation"
    )

    if correlation is None:
        return (
            f"{x} and {y} show a "
            f"{direction} relationship."
        )

    correlation_text = (
        f"{correlation:.3f}"
    )

    if strength >= 0.70:
        strength_text = "very strong"
    elif strength >= 0.50:
        strength_text = "moderate"
    elif strength >= 0.30:
        strength_text = "weak"
    else:
        strength_text = "very weak"

    if direction == "positive":
        movement_text = (
            f"As {x} increases, {y} also tends "
            f"to increase."
        )
    elif direction == "negative":
        movement_text = (
            f"As {x} increases, {y} tends "
            f"to decrease."
        )
    else:
        movement_text = (
            f"{x} and {y} do not show a clear "
            f"directional relationship."
        )

    return (
        f"{x} and {y} show a {strength_text} "
        f"{direction} association "
        f"(correlation: {correlation_text}). "
        f"{movement_text}"
    )


def explain_group_difference(pattern):
    """
    Generate a human-readable explanation
    for a group difference pattern.
    """

    dimension = pattern.get(
        "dimension"
    )

    measure = pattern.get(
        "measure"
    )

    highest_group = pattern.get(
        "highest_group"
    )

    highest_value = pattern.get(
        "highest_value"
    )

    lowest_group = pattern.get(
        "lowest_group"
    )

    lowest_value = pattern.get(
        "lowest_value"
    )

    if (
        highest_group is None
        or lowest_group is None
    ):
        return (
            f"{measure} varies across "
            f"{dimension}."
        )

    return (
        f"{measure} varies across "
        f"{dimension}. "
        f"{highest_group} has the highest "
        f"average ({highest_value:.2f}), "
        f"while {lowest_group} has the lowest "
        f"average ({lowest_value:.2f})."
    )


def explain_time_trend(pattern):
    """
    Generate a human-readable explanation
    for a time trend pattern.
    """

    measure = pattern.get(
        "measure"
    )

    trend = pattern.get(
        "trend",
        "stable"
    )

    change_pct = pattern.get(
        "change_pct",
        0
    )

    r2 = pattern.get(
        "r2",
        0
    )

    if trend == "increasing":
        direction_text = "increased"
    elif trend == "decreasing":
        direction_text = "decreased"
    else:
        direction_text = "remained relatively stable"

    return (
        f"{measure} {direction_text} over the "
        f"observed period, with an estimated "
        f"overall change of {change_pct:.2f}%. "
        f"The trend model has an R² of {r2:.3f}."
    )


def generate_explanation(pattern):
    """
    Generate a human-readable explanation
    based on the pattern type.
    """

    pattern_type = pattern.get(
        "pattern_type"
    )

    if pattern_type in [
        "strong_correlation",
        "moderate_correlation",
        "weak_correlation"
    ]:
        return explain_correlation(
            pattern
        )

    if pattern_type == "group_difference":
        return explain_group_difference(
            pattern
        )

    if pattern_type == "time_trend":
        return explain_time_trend(
            pattern
        )

    return (
        pattern.get(
            "description",
            "No explanation available."
        )
    )


def attach_explanations(patterns):
    """
    Attach a human-readable explanation
    to every pattern.
    """

    enriched_patterns = []

    for pattern in patterns:

        enriched_pattern = pattern.copy()

        enriched_pattern["explanation"] = (
            generate_explanation(
                enriched_pattern
            )
        )

        enriched_patterns.append(
            enriched_pattern
        )

    return enriched_patterns

