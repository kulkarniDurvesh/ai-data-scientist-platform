"""
Translate analysis results into ChartSpecs.

    insight (from core.eda_pipeline)        -> ChartSpec
    QueryPlan (from the NL query layer)     -> ChartSpec

Both mappers return None when no chart adds value.
"""

from __future__ import annotations

from typing import Any

from core.schema_inference import DatasetSchema

from .chart_schema import ChartSpec


CORRELATION_PATTERNS = {
    "strong_correlation",
    "moderate_correlation",
    "weak_correlation",
}

# How many groups an "argmax" style answer shows for context.
CONTEXT_LIMIT = 10


def insight_to_chart_spec(insight: dict[str, Any]) -> ChartSpec | None:
    pattern_type = insight.get("pattern_type")
    pattern = insight.get("pattern", {})

    if pattern_type in CORRELATION_PATTERNS:
        return ChartSpec(
            chart_type="SCATTER",
            x_column=pattern.get("x"),
            y_column=pattern.get("y"),
            aggregation="NONE",
        )

    if pattern_type == "group_difference":
        return ChartSpec(
            chart_type="BAR",
            x_column=pattern.get("dimension"),
            y_column=pattern.get("measure"),
            aggregation="AVERAGE",
        )

    if pattern_type == "time_trend":
        return ChartSpec(
            chart_type="LINE",
            x_column=pattern.get("time"),
            y_column=pattern.get("measure"),
            aggregation="AVERAGE",
            time_grain="M",
        )

    return None


def query_to_chart_spec(
    query_plan,
    schema: DatasetSchema,
) -> ChartSpec | None:

    operation = query_plan.operation
    column = query_plan.column
    filters = list(query_plan.filters or [])

    if operation == "CHART":
        return _requested_chart(query_plan, filters)

    if operation in {"TOP_N", "BOTTOM_N"} and query_plan.subject:
        return ChartSpec(
            chart_type="BAR",
            x_column=query_plan.subject,
            y_column=column,
            aggregation="SUM",
            filters=filters,
            sort="DESC" if operation == "TOP_N" else "ASC",
            limit=query_plan.limit,
        )

    if operation in {"ARGMAX", "ARGMIN"} and query_plan.subject:
        return ChartSpec(
            chart_type="BAR",
            x_column=query_plan.subject,
            y_column=column,
            aggregation="MAX" if operation == "ARGMAX" else "MIN",
            filters=filters,
            sort="DESC" if operation == "ARGMAX" else "ASC",
            limit=CONTEXT_LIMIT,
        )

    if operation in {"SUM", "AVERAGE", "MIN", "MAX"}:

        if query_plan.group_by:
            return ChartSpec(
                chart_type="BAR",
                x_column=query_plan.group_by,
                y_column=column,
                aggregation=operation,
                filters=filters,
            )

        if not schema.is_numeric(column):
            return None

        if schema.primary_time:
            return ChartSpec(
                chart_type="LINE",
                x_column=schema.primary_time,
                y_column=column,
                aggregation=operation,
                filters=filters,
                time_grain="AUTO",
            )

        return ChartSpec(
            chart_type="HISTOGRAM",
            x_column=column,
            filters=filters,
        )

    if operation == "COUNT":
        return _count_chart(column, filters, schema)

    if operation == "FILTER" and schema.role_of(column) == "measure":
        return ChartSpec(
            chart_type="HISTOGRAM",
            x_column=column,
            filters=filters,
        )

    return None


def _requested_chart(
    query_plan,
    filters: list[dict[str, Any]],
) -> ChartSpec | None:
    """The chart the question asked for ("pie chart of X by Y")."""

    chart_type = query_plan.chart_type or "BAR"
    column = query_plan.column
    group = query_plan.group_by

    if chart_type == "HISTOGRAM" or group is None:
        if column is None:
            return None
        if chart_type == "BOX":
            return ChartSpec(chart_type="BOX", y_column=column, filters=filters)
        return ChartSpec(chart_type="HISTOGRAM", x_column=column, filters=filters)

    if chart_type in ("BOX", "SCATTER"):
        return ChartSpec(
            chart_type=chart_type,
            x_column=group,
            y_column=column,
            aggregation="NONE",
            filters=filters,
        )

    return ChartSpec(
        chart_type=chart_type,
        x_column=group,
        y_column=column,
        aggregation=query_plan.aggregation or "COUNT",
        filters=filters,
        sort="DESC" if chart_type == "BAR" else None,
        time_grain=query_plan.time_grain,
    )


def _count_chart(
    column: str | None,
    filters: list[dict[str, Any]],
    schema: DatasetSchema,
) -> ChartSpec | None:
    """
    "How many X are in group G?" -> counts for every group, so the
    answer can be seen in context.
    """

    for index, filter_item in enumerate(filters):
        group = filter_item.get("column")

        if filter_item.get("operator") not in {"=", "=="}:
            continue

        if schema.role_of(group) not in {"dimension", "binary"}:
            continue

        if group == column:
            continue

        remaining = filters[:index] + filters[index + 1:]

        return ChartSpec(
            chart_type="BAR",
            x_column=group,
            y_column=column,
            aggregation="COUNT_DISTINCT" if column else "COUNT",
            filters=remaining,
            limit=CONTEXT_LIMIT * 2,
        )

    return None
