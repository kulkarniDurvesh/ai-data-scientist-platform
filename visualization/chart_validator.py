import pandas as pd

from .chart_schema import (
    NUMERIC_AGGREGATIONS,
    SUPPORTED_AGGREGATIONS,
    SUPPORTED_CHART_TYPES,
    SUPPORTED_FILTER_OPERATORS,
    SUPPORTED_SORTS,
    SUPPORTED_TIME_GRAINS,
    ChartSpec,
)


class ChartValidator:
    """
    Validates chart configurations against a DataFrame.

    This is the only place chart rules live: ChartEngine and
    ManualChartBuilder both delegate here.

    This class contains generic rules only.
    It does not know anything about a specific dataset.
    """

    def validate(
        self,
        dataframe: pd.DataFrame,
        chart_spec: ChartSpec,
    ) -> None:

        if dataframe is None:
            raise ValueError(
                "DataFrame cannot be None."
            )

        if not isinstance(
            dataframe,
            pd.DataFrame
        ):
            raise TypeError(
                "dataframe must be a pandas DataFrame."
            )

        if dataframe.empty:
            raise ValueError(
                "Cannot create a chart from an empty dataset."
            )

        if chart_spec is None:
            raise ValueError(
                "ChartSpec cannot be None."
            )

        chart_type = chart_spec.chart_type
        aggregation = chart_spec.aggregation or "NONE"

        if chart_type not in SUPPORTED_CHART_TYPES:
            raise ValueError(
                f"Unsupported chart type '{chart_type}'. "
                f"Supported types: {sorted(SUPPORTED_CHART_TYPES)}"
            )

        if aggregation not in SUPPORTED_AGGREGATIONS:
            raise ValueError(
                f"Unsupported aggregation '{aggregation}'. "
                f"Supported aggregations: "
                f"{sorted(SUPPORTED_AGGREGATIONS)}"
            )

        self._validate_filters(dataframe, chart_spec.filters)
        self._validate_options(dataframe, chart_spec)

        if chart_type == "HISTOGRAM":
            self._validate_histogram(dataframe, chart_spec, aggregation)

        elif chart_type == "BOX":
            self._validate_box(dataframe, chart_spec)

        elif chart_type == "SCATTER":
            self._validate_scatter(dataframe, chart_spec, aggregation)

        else:
            self._validate_grouped(dataframe, chart_spec, aggregation)

    # ======================================================
    # CHART TYPES
    # ======================================================

    def _validate_histogram(self, dataframe, spec, aggregation):

        self._require(spec.x_column, "Histogram requires a column.")
        self._validate_column(dataframe, spec.x_column)

        if not self._is_numeric_column(dataframe, spec.x_column):
            raise ValueError(
                "Histogram requires a numeric column."
            )

        if aggregation != "NONE":
            raise ValueError(
                "Histograms use raw values; set aggregation to NONE."
            )

    def _validate_box(self, dataframe, spec):

        self._require(
            spec.y_column,
            "Box plot requires a numeric Y column.",
        )
        self._validate_column(dataframe, spec.y_column)

        if not self._is_numeric_column(dataframe, spec.y_column):
            raise ValueError(
                "Box plot requires a numeric Y column."
            )

        if spec.x_column:
            self._validate_column(dataframe, spec.x_column)

            if spec.x_column == spec.y_column:
                raise ValueError(
                    "Box plot group and value columns must differ."
                )

    def _validate_scatter(self, dataframe, spec, aggregation):

        self._require(spec.x_column, "Scatter chart requires an X column.")
        self._require(spec.y_column, "Scatter chart requires a Y column.")

        self._validate_column(dataframe, spec.x_column)
        self._validate_column(dataframe, spec.y_column)

        for axis, column in (("X", spec.x_column), ("Y", spec.y_column)):
            if not self._is_numeric_column(dataframe, column):
                raise ValueError(
                    f"Scatter chart requires a numeric {axis} column."
                )

        if spec.x_column == spec.y_column:
            raise ValueError(
                "Scatter X and Y columns must differ."
            )

        if aggregation != "NONE":
            raise ValueError(
                "Scatter charts plot raw values; set aggregation to NONE."
            )

    def _validate_grouped(self, dataframe, spec, aggregation):
        """BAR / LINE / PIE."""

        chart_type = spec.chart_type

        self._require(
            spec.x_column,
            f"{chart_type} chart requires an X column.",
        )
        self._validate_column(dataframe, spec.x_column)

        if spec.y_column is not None:
            self._validate_column(dataframe, spec.y_column)

            if spec.y_column == spec.x_column:
                raise ValueError(
                    "X and Y columns must differ."
                )

        if aggregation == "COUNT":
            return

        if spec.y_column is None:
            raise ValueError(
                f"{aggregation} aggregation requires a Y column."
            )

        if aggregation == "COUNT_DISTINCT":
            return

        if not self._is_numeric_column(dataframe, spec.y_column):
            raise ValueError(
                f"Y column '{spec.y_column}' is not numeric; "
                f"use COUNT or COUNT_DISTINCT."
            )

        if aggregation == "NONE" and chart_type == "PIE":
            raise ValueError(
                "Pie charts require an aggregation."
            )

        if aggregation not in NUMERIC_AGGREGATIONS | {"NONE"}:
            raise ValueError(
                f"Unsupported aggregation '{aggregation}'."
            )

    # ======================================================
    # OPTIONS
    # ======================================================

    def _validate_options(self, dataframe, spec):

        if spec.color_column:
            self._validate_column(dataframe, spec.color_column)

            if spec.color_column in {spec.x_column, spec.y_column}:
                raise ValueError(
                    "Color/group column must differ from the X and "
                    "Y columns."
                )

            if spec.chart_type == "PIE":
                raise ValueError(
                    "Pie charts do not support a color/group column."
                )

        if spec.limit is not None:
            if not isinstance(spec.limit, int) or spec.limit <= 0:
                raise ValueError(
                    "Limit must be a positive whole number."
                )

        if spec.sort is not None and spec.sort not in SUPPORTED_SORTS:
            raise ValueError(
                f"Unsupported sort '{spec.sort}'. "
                f"Supported: {sorted(SUPPORTED_SORTS)}"
            )

        if spec.time_grain is not None:
            if spec.time_grain not in SUPPORTED_TIME_GRAINS:
                raise ValueError(
                    f"Unsupported time grain '{spec.time_grain}'. "
                    f"Supported: {sorted(SUPPORTED_TIME_GRAINS)}"
                )

    # ======================================================
    # COLUMN VALIDATION
    # ======================================================

    @staticmethod
    def _require(value, message: str) -> None:
        if not value:
            raise ValueError(message)

    @staticmethod
    def _validate_column(
        dataframe: pd.DataFrame,
        column: str
    ) -> None:

        if column not in dataframe.columns:
            raise ValueError(
                f"Column '{column}' does not exist in the dataset."
            )

    # ======================================================
    # FILTER VALIDATION
    # ======================================================

    def _validate_filters(
        self,
        dataframe: pd.DataFrame,
        filters: list | None
    ) -> None:

        for filter_item in filters or []:

            if not isinstance(filter_item, dict):
                raise ValueError(
                    "Each filter must be a dictionary."
                )

            column = filter_item.get("column")
            operator = filter_item.get("operator")

            self._validate_column(dataframe, column)

            if operator not in SUPPORTED_FILTER_OPERATORS:
                raise ValueError(
                    f"Unsupported filter operator '{operator}'."
                )

    # ======================================================
    # TYPE HELPERS
    # ======================================================

    @staticmethod
    def _is_numeric_column(
        dataframe: pd.DataFrame,
        column: str
    ) -> bool:

        series = dataframe[column]

        return (
            pd.api.types.is_numeric_dtype(series)
            and not pd.api.types.is_bool_dtype(series)
        )

    @staticmethod
    def is_datetime_column(
        dataframe: pd.DataFrame,
        column: str
    ) -> bool:

        return pd.api.types.is_datetime64_any_dtype(
            dataframe[column]
        )
