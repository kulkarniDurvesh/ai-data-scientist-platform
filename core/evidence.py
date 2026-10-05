
def generate_correlation_evidence(pattern):
    """
    Generate structured evidence for a correlation pattern.

    The function supports both the detailed pattern
    fields and the fallback correlation field.
    """

    correlation = pattern.get(
        "correlation"
    )

    strength = pattern.get(
        "strength"
    )

    pearson = pattern.get(
        "pearson"
    )

    spearman = pattern.get(
        "spearman"
    )

    sample_size = pattern.get(
        "sample_size"
    )

    # Fallback for patterns where the detailed
    # correlation metrics are not available.
    if pearson is None:
        pearson = correlation

    if strength is None and correlation is not None:
        strength = abs(
            correlation
        )

    return {
        "evidence_type": "correlation",
        "x": pattern.get("x"),
        "y": pattern.get("y"),
        "pearson": pearson,
        "spearman": spearman,
        "correlation": correlation,
        "strength": strength,
        "direction": pattern.get(
            "direction"
        ),
        "sample_size": sample_size
    }



def generate_group_difference_evidence(pattern):
    """
    Generate structured evidence for a dimension-measure
    group difference pattern.
    """

    return {
        "evidence_type": "group_difference",
        "dimension": pattern.get("dimension"),
        "measure": pattern.get("measure"),
        "highest_group": pattern.get("highest_group"),
        "highest_value": pattern.get("highest_value"),
        "lowest_group": pattern.get("lowest_group"),
        "lowest_value": pattern.get("lowest_value"),
        "spread_ratio": pattern.get("spread_ratio")
    }


def generate_time_trend_evidence(pattern):
    """
    Generate structured evidence for a time trend.
    """

    return {
        "evidence_type": "time_trend",
        "time": pattern.get("time"),
        "measure": pattern.get("measure"),
        "trend": pattern.get("trend"),
        "change_pct": pattern.get("change_pct"),
        "r2": pattern.get("r2"),
        "normalized_slope": pattern.get(
            "normalized_slope"
        )
    }


def generate_evidence(pattern):
    """
    Generate structured evidence based on
    the detected pattern type.
    """

    pattern_type = pattern.get(
        "pattern_type"
    )

    if pattern_type in [
        "strong_correlation",
        "moderate_correlation",
        "weak_correlation"
    ]:
        return generate_correlation_evidence(
            pattern
        )

    if pattern_type == "group_difference":
        return generate_group_difference_evidence(
            pattern
        )

    if pattern_type == "time_trend":
        return generate_time_trend_evidence(
            pattern
        )

    return {
        "evidence_type": "unknown"
    }


def attach_evidence(patterns):
    """
    Attach structured evidence to every detected pattern.

    The original pattern is copied so that this function
    does not modify the existing pattern objects.
    """

    enriched_patterns = []

    for pattern in patterns:

        enriched_pattern = pattern.copy()

        enriched_pattern["evidence"] = (
            generate_evidence(
                pattern
            )
        )

        enriched_patterns.append(
            enriched_pattern
        )

    return enriched_patterns




