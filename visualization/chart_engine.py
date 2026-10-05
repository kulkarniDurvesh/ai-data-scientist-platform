from __future__ import annotations

from typing import Any

import pandas as pd

from core.date_parts import DATE_PART_OPERATORS, date_part_mask
from core.schema_inference import suggest_time_grain

from .chart_schema import (
    SUPPORTED_AGGREGATIONS,
    SUPPORTED_CHART_TYPES,
    ChartResult,
    ChartSpec,
)
from .chart_validator import ChartValidator


MISSING_LABEL = "(missing)"


class ChartEngine:
    """
    Executes a ChartSpec against a pandas DataFrame.

    Responsibilities:
    - Validate chart configuration (via ChartValidator)
    - Apply filters
    - Bucket time values
    - Aggregate data when required
    - Sort and limit categories
    - Prepare chart-ready records
    - Return metadata for the visualization layer

    The engine does NOT decide which chart should be used.
    That responsibility belongs to ChartRecommender.
    """

    SUPPORTED_CHART_TYPES = SUPPORTED_CHART_TYPES
    SUPPORTED_AGGREGATIONS = SUPPORTED_AGGREGATIONS

    # Categorical axes longer than this are cut to the top N.
    DEFAULT_CATEGORY_LIMIT = 25

    # Series in a color split beyond this are folded into "Other".
    MAX_COLOR_SERIES = 8

    # Scatter plots beyond this are randomly sampled.
    MAX_SCATTER_POINTS = 5000

    _PANDAS_AGGREGATIONS = {
        "SUM": "sum",
        "AVERAGE": "mean",
        "MIN": "min",
        "MAX": "max",
    }

    def __init__(self, validator: ChartValidator | None = None):
        self.validator = validator or ChartValidator()

    # ============================================================
    # Create Chart
    # ============================================================

    def create_chart(
        self,
        dataframe: pd.DataFrame,
        chart_spec: ChartSpec,
    ) -> ChartResult:

        self.validator.validate(dataframe, chart_spec)

        chart_type = chart_spec.chart_type
        aggregation = chart_spec.aggregation or "NONE"

        filtered = self._apply_filters(dataframe, chart_spec.filters)

        if filtered.empty:
            raise ValueError(
                "No data remains after applying filters."
            )

        metadata: dict[str, Any] = {
            "aggregation": aggregation,
            "chart_type": chart_type,
            "source_rows": len(filtered),
        }

        if chart_type in {"BAR", "LINE", "PIE"}:
            data, value_column = self._prepare_grouped(
                filtered, chart_spec, aggregation, metadata,
            )

        elif chart_type == "SCATTER":
            data = self._prepare_scatter(filtered, chart_spec, metadata)
            value_column = chart_spec.y_column

        elif chart_type == "HISTOGRAM":
            data = self._prepare_raw(
                filtered, [chart_spec.x_column], chart_spec.color_column,
            )
            value_column = None

        elif chart_type == "BOX":
            data = self._prepare_box(filtered, chart_spec, metadata)
            value_column = chart_spec.y_column

        else:
            raise ValueError(
                f"Unsupported chart type: '{chart_type}'"
            )

        if data.empty:
            raise ValueError(
                "No chart data available after processing."
            )

        metadata["row_count"] = len(data)
        metadata["value_column"] = value_column

        return ChartResult(
            chart_spec=chart_spec,
            data=self._clean_result(data).to_dict(orient="records"),
            x_column=chart_spec.x_column,
            y_column=value_column,
            metadata=metadata,
        )

    def create(
        self,
        dataframe: pd.DataFrame,
        chart_spec: ChartSpec,
    ) -> ChartResult:

        return self.create_chart(dataframe, chart_spec)

    # ============================================================
    # Filters
    # ============================================================

    def _apply_filters(
        self,
        dataframe: pd.DataFrame,
        filters: list[dict[str, Any]] | None,
    ) -> pd.DataFrame:

        if not filters:
            return dataframe

        mask = pd.Series(True, index=dataframe.index)

        for filter_item in filters:
            column = filter_item.get("column")
            operator = filter_item.get("operator")
            value = filter_item.get("value")

            series = dataframe[column]

            if operator in DATE_PART_OPERATORS:
                mask &= date_part_mask(series, operator, value)
            elif operator == ">":
                mask &= series > value
            elif operator == ">=":
                mask &= series >= value
            elif operator == "<":
                mask &= series < value
            elif operator == "<=":
                mask &= series <= value
            elif operator in {"=", "=="}:
                mask &= series == value
            elif operator in {"!=", "<>"}:
                mask &= series != value
            elif operator == "in":
                mask &= series.isin(self._as_list(value))
            elif operator == "not in":
                mask &= ~series.isin(self._as_list(value))
            elif operator == "contains":
                mask &= series.astype(str).str.contains(
                    str(value), case=False, na=False, regex=False,
                )
            else:
                raise ValueError(
                    f"Unsupported filter operator: '{operator}'"
                )

        return dataframe.loc[mask]

    @staticmethod
    def _as_list(value: Any) -> list[Any]:
        if isinstance(value, (list, tuple, set)):
            return list(value)
        return [value]

    # ============================================================
    # BAR / LINE / PIE
    # ============================================================

    def _prepare_grouped(
        self,
        dataframe: pd.DataFrame,
        spec: ChartSpec,
        aggregation: str,
        metadata: dict[str, Any],
    ) -> tuple[pd.DataFrame, str]:

        x_column = spec.x_column
        y_column = spec.y_column
        color_column = spec.color_column

        frame = dataframe.copy()

        x_is_time = pd.api.types.is_datetime64_any_dtype(frame[x_column])

        if x_is_time:
            grain = self._resolve_time_grain(frame[x_column], spec)

            if grain:
                frame[x_column] = self._bucket_time(frame[x_column], grain)
                metadata["time_grain"] = grain
        else:
            frame[x_column] = self._label_missing(frame[x_column])

        keys = [x_column]

        if color_column:
            frame[color_column] = self._fold_series(
                self._label_missing(frame[color_column])
            )
            keys.append(color_column)

        # Raw values: no aggregation.
        if aggregation == "NONE":
            columns = keys + [y_column]
            result = frame[columns].dropna(subset=[y_column])
            value_column = y_column

        else:
            result, value_column = self._aggregate(
                frame, keys, y_column, aggregation,
            )

        result = self._sort_and_limit(
            result, spec, x_column, value_column, x_is_time, metadata,
        )

        return result, value_column

    def _aggregate(
        self,
        frame: pd.DataFrame,
        keys: list[str],
        y_column: str | None,
        aggregation: str,
    ) -> tuple[pd.DataFrame, str]:

        grouped = frame.groupby(keys, dropna=False, observed=True)

        if aggregation == "COUNT":
            value_column = "Count"
            result = grouped.size().reset_index(name=value_column)
            return result, value_column

        value_column = y_column

        if aggregation == "COUNT_DISTINCT":
            result = (
                grouped[y_column]
                .nunique(dropna=True)
                .reset_index(name=value_column)
            )
            return result, value_column

        if aggregation.endswith("_DISTINCT"):
            base = aggregation.replace("_DISTINCT", "")
            function = self._PANDAS_AGGREGATIONS[base]

            result = (
                grouped[y_column]
                .agg(lambda series: series.dropna().drop_duplicates().agg(function))
                .reset_index(name=value_column)
            )
            return result, value_column

        function = self._PANDAS_AGGREGATIONS.get(aggregation)

        if function is None:
            raise ValueError(
                f"Unsupported aggregation: '{aggregation}'"
            )

        result = (
            grouped[y_column]
            .agg(function)
            .reset_index(name=value_column)
        )

        return result, value_column

    def _sort_and_limit(
        self,
        result: pd.DataFrame,
        spec: ChartSpec,
        x_column: str,
        value_column: str,
        x_is_time: bool,
        metadata: dict[str, Any],
    ) -> pd.DataFrame:

        x_is_numeric = pd.api.types.is_numeric_dtype(result[x_column])

        sort = spec.sort

        if sort is None:
            # Time and numeric axes keep their natural order; categories
            # are ordered by value so the largest bar comes first.
            ordered_axis = x_is_time or x_is_numeric or spec.chart_type == "LINE"
            sort = "X" if ordered_axis else "DESC"

        # Rank categories by their total value across all series.
        totals = (
            result.groupby(x_column, dropna=False, observed=True)[value_column]
            .sum(min_count=1)
        )

        if sort == "X":
            order = sorted(totals.index, key=self._sort_key)
        else:
            order = (
                totals.sort_values(ascending=(sort == "ASC"))
                .index
                .tolist()
            )

        limit = spec.limit

        if (
            limit is None
            and sort != "X"
            and len(order) > self.DEFAULT_CATEGORY_LIMIT
        ):
            limit = self.DEFAULT_CATEGORY_LIMIT

        if limit is not None and len(order) > limit:
            metadata["truncated_from"] = len(order)
            order = order[:limit]

        metadata["category_count"] = len(order)

        position = {value: index for index, value in enumerate(order)}

        result = result[result[x_column].isin(order)].copy()
        result["_order"] = result[x_column].map(position)

        sort_columns = ["_order"]

        if spec.color_column and spec.color_column in result.columns:
            sort_columns.append(spec.color_column)

        result = result.sort_values(sort_columns, kind="stable")

        return result.drop(columns="_order").reset_index(drop=True)

    @staticmethod
    def _sort_key(value: Any) -> tuple:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return (2, "")
        if isinstance(value, (int, float, pd.Timestamp)):
            return (0, value)
        return (1, str(value))

    # ============================================================
    # Time handling
    # ============================================================

    @staticmethod
    def _resolve_time_grain(series: pd.Series, spec: ChartSpec) -> str | None:
        if spec.time_grain in (None, "AUTO"):
            return suggest_time_grain(series)
        return spec.time_grain

    @staticmethod
    def _bucket_time(series: pd.Series, grain: str) -> pd.Series:
        period = {"D": "D", "W": "W", "M": "M", "Q": "Q", "Y": "Y"}[grain]
        return series.dt.to_period(period).dt.start_time

    # ============================================================
    # Categorical helpers
    # ============================================================

    @staticmethod
    def _label_missing(series: pd.Series) -> pd.Series:
        if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_datetime64_any_dtype(series):
            return series

        if not series.isna().any():
            return series

        return series.astype(object).where(series.notna(), MISSING_LABEL)

    def _fold_series(self, series: pd.Series) -> pd.Series:
        """Keep the most frequent series and fold the rest into 'Other'."""

        counts = series.value_counts(dropna=False)

        if len(counts) <= self.MAX_COLOR_SERIES:
            return series

        keep = set(counts.index[: self.MAX_COLOR_SERIES - 1])

        return series.astype(object).where(series.isin(keep), "Other")

    # ============================================================
    # Scatter
    # ============================================================

    def _prepare_scatter(
        self,
        dataframe: pd.DataFrame,
        spec: ChartSpec,
        metadata: dict[str, Any],
    ) -> pd.DataFrame:

        result = self._prepare_raw(
            dataframe, [spec.x_column, spec.y_column], spec.color_column,
        )

        if len(result) > self.MAX_SCATTER_POINTS:
            metadata["sampled_from"] = len(result)
            result = result.sample(self.MAX_SCATTER_POINTS, random_state=0)

        return result

    # ============================================================
    # Histogram / raw values
    # ============================================================

    def _prepare_raw(
        self,
        dataframe: pd.DataFrame,
        value_columns: list[str],
        color_column: str | None,
    ) -> pd.DataFrame:

        columns = list(value_columns)

        if color_column:
            columns.append(color_column)

        result = dataframe[columns].dropna(subset=value_columns).copy()

        if color_column:
            result[color_column] = self._fold_series(
                self._label_missing(result[color_column])
            )

        return result

    # ============================================================
    # Box Plot
    # ============================================================

    def _prepare_box(
        self,
        dataframe: pd.DataFrame,
        spec: ChartSpec,
        metadata: dict[str, Any],
    ) -> pd.DataFrame:

        if not spec.x_column:
            return self._prepare_raw(dataframe, [spec.y_column], None)

        result = dataframe[[spec.x_column, spec.y_column]].dropna(
            subset=[spec.y_column]
        ).copy()

        result[spec.x_column] = self._label_missing(result[spec.x_column])

        counts = result[spec.x_column].value_counts()
        limit = spec.limit or self.DEFAULT_CATEGORY_LIMIT

        if len(counts) > limit:
            metadata["truncated_from"] = len(counts)
            result = result[result[spec.x_column].isin(counts.index[:limit])]

        return result

    # ============================================================
    # Utilities
    # ============================================================

    def _clean_result(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:

        # Replace NaN / inf values so the result
        # can safely be serialized to JSON.

        result = dataframe.replace(
            [float("inf"), float("-inf")],
            pd.NA,
        )

        return result.astype(object).where(
            pd.notnull(result),
            None,
        )

    def get_supported_chart_types(
        self,
    ) -> list[str]:
        return sorted(self.SUPPORTED_CHART_TYPES)

    def get_supported_aggregations(
        self,
    ) -> list[str]:
        return sorted(self.SUPPORTED_AGGREGATIONS)
