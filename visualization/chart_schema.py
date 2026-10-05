from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any

from core.date_parts import DATE_PART_OPERATORS
from core.schema_inference import humanize


SUPPORTED_CHART_TYPES = {
    "BAR",
    "LINE",
    "SCATTER",
    "PIE",
    "HISTOGRAM",
    "BOX",
}


SUPPORTED_AGGREGATIONS = {
    "SUM",
    "SUM_DISTINCT",
    "AVERAGE",
    "AVERAGE_DISTINCT",
    "COUNT",
    "COUNT_DISTINCT",
    "MIN",
    "MIN_DISTINCT",
    "MAX",
    "MAX_DISTINCT",
    "NONE",
}

NUMERIC_AGGREGATIONS = {
    "SUM",
    "SUM_DISTINCT",
    "AVERAGE",
    "AVERAGE_DISTINCT",
    "MIN",
    "MIN_DISTINCT",
    "MAX",
    "MAX_DISTINCT",
}

SUPPORTED_FILTER_OPERATORS = {
    "=", "==", "!=", "<>", ">", ">=", "<", "<=", "in", "not in", "contains",
} | DATE_PART_OPERATORS

# None = keep raw values, AUTO = let the engine pick a readable grain.
SUPPORTED_TIME_GRAINS = {"AUTO", "D", "W", "M", "Q", "Y"}

SUPPORTED_SORTS = {"DESC", "ASC", "X"}


@dataclass
class ChartSpec:
    """
    Describes how a chart should be created.

    This class is dataset-independent.

    Example:

        ChartSpec(
            chart_type="BAR",
            x_column="<category column>",
            y_column="<numeric column>",
            aggregation="SUM",
            color_column="<optional second category>",
            limit=10,
        )
    """

    chart_type: str

    x_column: str | None = None
    y_column: str | None = None

    aggregation: str = "NONE"

    filters: list[dict[str, Any]] = field(
        default_factory=list
    )

    title: str | None = None
    reason: str | None = None

    # Optional series split (legend).
    color_column: str | None = None

    # Keep only the first N categories after sorting.
    limit: int | None = None

    # DESC / ASC by value, X by category/time; None = engine default.
    sort: str | None = None

    # Bucket datetime X values: AUTO, D, W, M, Q, Y.
    time_grain: str | None = None

    def __post_init__(self):
        self.chart_type = str(self.chart_type).upper()

        self.aggregation = (
            str(self.aggregation).upper()
            if self.aggregation
            else "NONE"
        )

        if self.sort is not None:
            self.sort = str(self.sort).upper()

        if self.time_grain is not None:
            self.time_grain = str(self.time_grain).upper()

        if not self.title:
            self.title = default_title(self)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chart_type": self.chart_type,
            "x_column": self.x_column,
            "y_column": self.y_column,
            "aggregation": self.aggregation,
            "filters": self.filters,
            "title": self.title,
            "reason": self.reason,
            "color_column": self.color_column,
            "limit": self.limit,
            "sort": self.sort,
            "time_grain": self.time_grain,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChartSpec":
        allowed = {item.name for item in fields(cls)}
        return cls(**{
            key: value
            for key, value in data.items()
            if key in allowed
        })


@dataclass
class ChartRecommendation:
    """
    Represents an automatically generated
    visualization recommendation.
    """

    chart_spec: ChartSpec

    confidence: float

    explanation: str

    category: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "chart_spec": self.chart_spec.to_dict(),
            "confidence": self.confidence,
            "explanation": self.explanation,
            "category": self.category,
        }


@dataclass
class ChartResult:
    """
    Result returned by ChartEngine after preparing
    chart-ready data.
    """

    chart_spec: ChartSpec

    data: list[dict[str, Any]]

    x_column: str | None = None
    y_column: str | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "chart_spec": self.chart_spec.to_dict(),
            "x_column": self.x_column,
            "y_column": self.y_column,
            "data": self.data,
            "metadata": self.metadata,
        }


# ----------------------------------------------------------------------
# Labels
# ----------------------------------------------------------------------

_AGGREGATION_WORDS = {
    "SUM": "Total",
    "SUM_DISTINCT": "Total distinct",
    "AVERAGE": "Average",
    "AVERAGE_DISTINCT": "Average distinct",
    "MIN": "Minimum",
    "MIN_DISTINCT": "Minimum",
    "MAX": "Maximum",
    "MAX_DISTINCT": "Maximum",
}


def value_label(aggregation: str | None, y_column: str | None) -> str:
    """Axis label for the plotted value: 'Average Order Rate', 'Count'."""

    aggregation = (aggregation or "NONE").upper()

    if aggregation == "COUNT":
        return "Count"

    if aggregation == "COUNT_DISTINCT":
        return f"Distinct {humanize(y_column)}"

    label = humanize(y_column)

    word = _AGGREGATION_WORDS.get(aggregation)

    if word is None:
        return label

    # Avoid "Total Total Visits" / "Average Avg Price".
    first_word = label.split(" ")[0].lower() if label else ""

    if first_word in {"total", "sum", "average", "avg", "mean"}:
        return label

    return f"{word} {label}"


def default_title(spec: ChartSpec) -> str:
    chart_type = spec.chart_type
    x_label = humanize(spec.x_column)
    y_label = humanize(spec.y_column)

    if chart_type == "HISTOGRAM":
        return f"Distribution of {x_label}"

    if chart_type == "BOX":
        if spec.x_column:
            return f"{y_label} by {x_label}"
        return f"Spread of {y_label}"

    if chart_type == "SCATTER":
        return f"{y_label} vs {x_label}"

    title = f"{value_label(spec.aggregation, spec.y_column)} by {x_label}"

    if spec.color_column:
        title += f" and {humanize(spec.color_column)}"

    return title
