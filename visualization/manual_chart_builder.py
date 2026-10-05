from __future__ import annotations

from typing import Any, Optional

import pandas as pd

from core.schema_inference import DatasetSchema, infer_schema

from .chart_recommender import ChartRecommender
from .chart_schema import (
    SUPPORTED_AGGREGATIONS,
    SUPPORTED_CHART_TYPES,
    ChartSpec,
)
from .chart_validator import ChartValidator


AUTO = "AUTO"


class ManualChartBuilder:
    """
    Builds ChartSpec objects from user-selected chart configuration.

    Responsibilities:
        - Resolve AUTO chart type / aggregation from the dataset schema
        - Create ChartSpec
        - Validate it through the shared ChartValidator

    It does NOT:
        - Execute queries
        - Transform dataframe data
        - Render charts
    """

    SUPPORTED_CHART_TYPES = SUPPORTED_CHART_TYPES
    SUPPORTED_AGGREGATIONS = SUPPORTED_AGGREGATIONS

    def __init__(
        self,
        validator: Optional[ChartValidator] = None,
        recommender: Optional[ChartRecommender] = None,
    ):
        self.validator = validator or ChartValidator()
        self.recommender = recommender or ChartRecommender()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
        dataframe: pd.DataFrame,
        chart_type: str | None = AUTO,
        x_column: Optional[str] = None,
        y_column: Optional[str] = None,
        aggregation: Optional[str] = AUTO,
        color_column: Optional[str] = None,
        limit: Optional[int] = None,
        filters: Optional[list[dict[str, Any]]] = None,
        time_grain: Optional[str] = None,
        schema: DatasetSchema | None = None,
    ) -> ChartSpec:
        """
        Build and validate a ChartSpec from manual user selections.

        chart_type:
            AUTO, BAR, LINE, SCATTER, HISTOGRAM, BOX or PIE.

        aggregation:
            AUTO, SUM, AVERAGE, COUNT, COUNT_DISTINCT, MIN, MAX, NONE
            (and their *_DISTINCT variants).
        """

        if dataframe is None or dataframe.empty:
            raise ValueError("Dataset is empty.")

        schema = schema or infer_schema(dataframe)

        chart_type = self._normalize(chart_type) or AUTO
        aggregation = self._normalize(aggregation) or AUTO

        if aggregation == "AVG":
            aggregation = "AVERAGE"

        if not x_column and not y_column:
            raise ValueError("Select at least one column.")

        if chart_type == AUTO:
            recommendation = self.recommender.recommend(
                dataframe,
                x_column=x_column,
                y_column=y_column,
                schema=schema,
            )
            base = recommendation.chart_spec
            chart_type = base.chart_type
            x_column, y_column = base.x_column, base.y_column

            if aggregation == AUTO:
                aggregation = base.aggregation

        if chart_type not in self.SUPPORTED_CHART_TYPES:
            raise ValueError(
                f"Unsupported chart type '{chart_type}'."
            )

        x_column, y_column = self._arrange_columns(
            chart_type, x_column, y_column,
        )

        aggregation = self._resolve_aggregation(
            schema, chart_type, y_column, aggregation,
        )

        if (
            time_grain is None
            and x_column
            and schema.role_of(x_column) == "time"
            and chart_type in {"BAR", "LINE"}
        ):
            time_grain = AUTO

        spec = ChartSpec(
            chart_type=chart_type,
            x_column=x_column,
            y_column=y_column,
            aggregation=aggregation,
            color_column=color_column or None,
            limit=int(limit) if limit else None,
            filters=list(filters or []),
            time_grain=time_grain,
        )

        self.validator.validate(dataframe, spec)

        return spec

    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------

    @staticmethod
    def _arrange_columns(chart_type, x_column, y_column):
        """Single-column charts accept the column in either slot."""

        if chart_type == "HISTOGRAM":
            return x_column or y_column, None

        if chart_type == "BOX" and y_column is None:
            return None, x_column

        return x_column, y_column

    def _resolve_aggregation(
        self,
        schema: DatasetSchema,
        chart_type: str,
        y_column: Optional[str],
        aggregation: str,
    ) -> str:

        # Scatter, histogram and box plots work with raw observations.
        if chart_type in {"SCATTER", "HISTOGRAM", "BOX"}:
            return "NONE"

        if aggregation != AUTO:
            if aggregation not in self.SUPPORTED_AGGREGATIONS:
                raise ValueError(
                    f"Unsupported aggregation '{aggregation}'."
                )
            return aggregation

        if y_column is None:
            return "COUNT"

        return ChartRecommender.default_aggregation(schema, y_column)

    @staticmethod
    def _normalize(value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip().upper()
        return text or None
