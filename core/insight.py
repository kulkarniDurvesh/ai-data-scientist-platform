def build_insight(pattern):
    """
    Build a clean final insight object from an enriched pattern.

    The input pattern is expected to contain:
    - pattern information
    - evidence
    - explanation
    - interestingness score
    """

    insight = {
        "pattern_type": pattern.get(
            "pattern_type"
        ),

        "pattern": {},

        "description": pattern.get(
            "description",
            ""
        ),

        "evidence": pattern.get(
            "evidence",
            {}
        ),

        "explanation": pattern.get(
            "explanation",
            ""
        ),

        "interestingness_score": pattern.get(
            "interestingness_score",
            0
        )
    }

    pattern_type = pattern.get(
        "pattern_type"
    )

    if pattern_type in [
        "strong_correlation",
        "moderate_correlation",
        "weak_correlation"
    ]:
        insight["pattern"] = {
            "x": pattern.get("x"),
            "y": pattern.get("y"),
            "direction": pattern.get(
                "direction"
            ),
            "correlation": pattern.get(
                "correlation"
            )
        }

    elif pattern_type == "group_difference":
        insight["pattern"] = {
            "dimension": pattern.get(
                "dimension"
            ),
            "measure": pattern.get(
                "measure"
            ),
            "highest_group": pattern.get(
                "highest_group"
            ),
            "lowest_group": pattern.get(
                "lowest_group"
            ),
            "highest_value": pattern.get(
                "highest_value"
            ),
            "lowest_value": pattern.get(
                "lowest_value"
            ),
            "spread_ratio": pattern.get(
                "spread_ratio"
            )
        }

    elif pattern_type == "time_trend":
        insight["pattern"] = {
            "time": pattern.get(
                "time"
            ),
            "measure": pattern.get(
                "measure"
            ),
            "trend": pattern.get(
                "trend"
            ),
            "change_pct": pattern.get(
                "change_pct"
            ),
            "r2": pattern.get(
                "r2"
            ),
            "normalized_slope": pattern.get(
                "normalized_slope"
            )
        }

    return insight


def build_insights(patterns):
    """
    Convert enriched patterns into final insight objects.
    """

    insights = []

    for pattern in patterns:
        insights.append(
            build_insight(pattern)
        )

    return insights



    