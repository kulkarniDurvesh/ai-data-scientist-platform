
import pandas as pd

from core.date_parts import DATE_PART_OPERATORS, as_datetime, date_part_mask
from core.schema_inference import (
    ENTITY_KEY_TOKENS,
    is_identifier,
    name_tokens,
    same_word,
)


# Descriptive columns shown next to each entity in a list answer.
MAX_LIST_ATTRIBUTES = 4
# Columns with longer average text are notes, not attributes.
MAX_ATTRIBUTE_TEXT_LENGTH = 40


class QueryEngine:
    """
    Executes a QueryPlan against a Pandas DataFrame.

    Supported operations:

        COUNT
        SUM
        AVERAGE
        MIN
        MAX
        FILTER   (distinct list for entity / text columns)
        CHART    (grouped values behind a requested chart)

    Supported operators:

        =
        >
        <
        >=
        <=
        !=
    """

    def __init__(self, dataframe, schema=None):

        self.df = dataframe

        self.schema = schema

        self._attribute_cache: dict[str, list[str]] = {}

    # ---------------------------------------------------------
    # PUBLIC API
    # ---------------------------------------------------------

    def execute(self, query_plan):

        if query_plan is None:
            raise ValueError(
                "Query plan cannot be None."
            )

        operation = query_plan.operation
        column = query_plan.column
        filters = query_plan.filters

        # -----------------------------------------------------
        # Apply filters first.
        # -----------------------------------------------------

        filtered_df = self._apply_filters(
            self.df,
            filters
        )

        # -----------------------------------------------------
        # Execute operation.
        # -----------------------------------------------------

        if operation == "COUNT":

            return self._count(
                filtered_df,
                column
            )

        if operation == "SUM":

            if query_plan.group_by is not None:
                return self._grouped_sum(
                    filtered_df,
                    column,
                    query_plan.group_by
                )

            return self._sum(
                filtered_df,
                column
            )

        if operation == "AVERAGE":

            if query_plan.group_by is not None:
                return self._grouped_average(
                    filtered_df,
                    column,
                    query_plan.group_by
                )

            return self._average(
                filtered_df,
                column
            )

        if operation == "MIN":

            if query_plan.group_by is not None:
                return self._grouped_minimum(
                    filtered_df,
                    column,
                    query_plan.group_by
                )

            return self._minimum(
                filtered_df,
                column
            )

        if operation == "MAX":

            if query_plan.group_by is not None:
                return self._grouped_maximum(
                    filtered_df,
                    column,
                    query_plan.group_by
                )

            return self._maximum(
                filtered_df,
                column
            )

        if operation == "FILTER":

            return self._filter(
                filtered_df,
                column
            )

        if operation == "CHART":

            return self._chart_table(
                filtered_df,
                query_plan
            )

        if operation == "ARGMAX":
            return self._argmax(
                filtered_df,
                column,
                query_plan.subject
            )

        if operation == "ARGMIN":
            return self._argmin(
                filtered_df,
                column,
                query_plan.subject
            )

        if operation == "TOP_N":
            return self._top_n(
                filtered_df,
                column,
                query_plan.subject,
                query_plan.limit
            )


        if operation == "BOTTOM_N":
            return self._bottom_n(
                filtered_df,
                column,
                query_plan.subject,
                query_plan.limit
            )

        raise ValueError(
            f"Unsupported query operation: "
            f"'{operation}'"
        )

    # ---------------------------------------------------------
    # FILTER PROCESSING
    # ---------------------------------------------------------

    def _apply_filters(
        self,
        dataframe: pd.DataFrame,
        filters: list[dict]
    ) -> pd.DataFrame:

        result = dataframe.copy()

        for filter_item in filters:

            column = filter_item["column"]
            operator = filter_item["operator"]
            value = filter_item["value"]

            if column not in result.columns:

                raise ValueError(
                    f"Filter column '{column}' "
                    f"does not exist in the dataset."
                )

            result = self._apply_single_filter(
                result,
                column,
                operator,
                value
            )

        return result

    def _apply_single_filter(
        self,
        dataframe: pd.DataFrame,
        column: str,
        operator: str,
        value
    ) -> pd.DataFrame:

        series = dataframe[column]

        if operator in DATE_PART_OPERATORS:

            mask = date_part_mask(series, operator, value)

        elif operator == "=":

            mask = series == value

        elif operator == "!=":

            mask = series != value

        elif operator == ">":

            mask = series > value

        elif operator == "<":

            mask = series < value

        elif operator == ">=":

            mask = series >= value

        elif operator == "<=":

            mask = series <= value

        else:

            raise ValueError(
                f"Unsupported filter operator: "
                f"'{operator}'"
            )

        return dataframe.loc[mask]

    # ---------------------------------------------------------
    # COUNT
    # ---------------------------------------------------------

    def _count(
        self,
        dataframe: pd.DataFrame,
        column: str | None
    ) -> int:

        if column is None:

            return len(dataframe)

        return int(
            dataframe[column]
            .dropna()
            .nunique()
        )

    # ---------------------------------------------------------
    # SUM
    # ---------------------------------------------------------

    def _sum(
        self,
        dataframe: pd.DataFrame,
        column: str
    ) -> float:

        self._validate_numeric_column(
            dataframe,
            column
        )

        return float(
            dataframe[column]
            .sum()
        )

    # ---------------------------------------------------------
    # AVERAGE
    # ---------------------------------------------------------

    def _average(
        self,
        dataframe: pd.DataFrame,
        column: str
    ) -> float:

        self._validate_numeric_column(
            dataframe,
            column
        )

        if dataframe[column].dropna().empty:

            return 0.0

        return float(
            dataframe[column]
            .mean()
        )

    # ---------------------------------------------------------
    # MINIMUM
    # ---------------------------------------------------------

    def _minimum(
        self,
        dataframe: pd.DataFrame,
        column: str
    ):

        if dataframe[column].dropna().empty:

            return None

        return dataframe[column].min()

    # ---------------------------------------------------------
    # MAXIMUM
    # ---------------------------------------------------------

    def _maximum(
        self,
        dataframe: pd.DataFrame,
        column: str
    ):

        if dataframe[column].dropna().empty:

            return None

        return dataframe[column].max()

    # ---------------------------------------------------------
    # FILTER
    # ---------------------------------------------------------

    def _filter(
        self,
        dataframe: pd.DataFrame,
        column: str | None
    ):

        if column is None:

            return dataframe

        if self._is_measure(column):

            return dataframe[
                [column]
            ].copy()

        return self._distinct_list(
            dataframe,
            column
        )

    # ---------------------------------------------------------
    # DISTINCT LIST
    # ---------------------------------------------------------

    def _is_measure(self, column: str) -> bool:
        """Numeric and not a key: its rows are values, not entities."""

        return (
            self.df[column].dtype.kind in "biufc"
            and not is_identifier(column, self.schema)
        )

    def _distinct_list(
        self,
        dataframe: pd.DataFrame,
        column: str
    ) -> pd.DataFrame:
        """
        One row per distinct value of the entity column, with the
        descriptive columns that belong to it (name, category, ...).
        """

        attributes = self._entity_attributes(column)

        result = (
            dataframe[[column] + attributes]
            .dropna(subset=[column])
            .drop_duplicates(subset=[column])
            .sort_values(column)
            .reset_index(drop=True)
        )

        result.attrs["distinct_entity"] = column

        return result

    def _entity_attributes(self, column: str) -> list[str]:
        """
        Text columns with a single value per entity across the whole
        dataset, e.g. DoctorId -> DoctorName, Specialty, Territory.
        """

        cache = self._attribute_cache

        if column in cache:
            return cache[column]

        candidates = []

        for other in self.df.columns:

            if other == column:
                continue

            series = self.df[other]

            if series.dtype.kind in "biufcmM":
                continue

            text = series.dropna().astype(str)

            if text.empty or text.str.len().mean() > MAX_ATTRIBUTE_TEXT_LENGTH:
                continue

            # The same value on every row says nothing about an entity.
            if text.nunique() <= 1:
                continue

            candidates.append(other)

        if not candidates:
            cache[column] = []
            return []

        per_entity = (
            self.df[[column] + candidates]
            .dropna(subset=[column])
            .groupby(column)[candidates]
            .nunique()
            .max()
        )

        dependent = [
            other
            for other in candidates
            if per_entity.get(other, 2) <= 1
        ]

        entity_tokens = set(name_tokens(column)) - ENTITY_KEY_TOKENS

        # Columns named after the entity come first, most distinctive
        # first (DoctorName before Doctors Territory); then the other
        # attributes, broadest categories first.
        def rank(other: str) -> tuple[bool, int]:
            shares_name = any(
                same_word(token, entity_token)
                for token in name_tokens(other)
                for entity_token in entity_tokens
            )
            unique = self.df[other].nunique()
            return (not shares_name, -unique if shares_name else unique)

        cache[column] = sorted(dependent, key=rank)[:MAX_LIST_ATTRIBUTES]

        return cache[column]

    # ---------------------------------------------------------
    # CHART TABLE
    # ---------------------------------------------------------

    def _chart_table(
        self,
        dataframe: pd.DataFrame,
        query_plan
    ) -> pd.DataFrame:
        """
        The numbers behind a requested chart: one row per group,
        largest first (time groups stay in time order).
        """

        group = query_plan.group_by
        column = query_plan.column
        aggregation = query_plan.aggregation or "COUNT"

        if group is None:

            if column is None or not self._is_measure(column):
                raise ValueError(
                    "This chart needs a numeric column to show."
                )

            summary = dataframe[column].describe()

            return summary.rename(column).reset_index().rename(
                columns={"index": "Statistic"}
            )

        keys = dataframe[group]
        is_time = pd.api.types.is_datetime64_any_dtype(keys)

        if is_time and query_plan.time_grain:
            keys = (
                as_datetime(keys)
                .dt.to_period(query_plan.time_grain)
                .dt.start_time
                .rename(group)
            )

        grouped = dataframe.groupby(keys)

        if aggregation == "COUNT":
            values = grouped.size()
            label = "Count"
        elif aggregation == "COUNT_DISTINCT":
            values = grouped[column].nunique()
            label = f"{column} (distinct)"
        elif aggregation == "AVERAGE":
            values = grouped[column].mean()
            label = f"Average {column}"
        elif aggregation == "MIN":
            values = grouped[column].min()
            label = f"Min {column}"
        elif aggregation == "MAX":
            values = grouped[column].max()
            label = f"Max {column}"
        else:
            values = grouped[column].sum()
            label = column

        result = values.rename(label).reset_index()

        if is_time:
            result = result.sort_values(group)
        else:
            result = result.sort_values(label, ascending=False)

        result = result.reset_index(drop=True)
        result.attrs["value_column"] = label

        return result

    # ---------------------------------------------------------
    # VALIDATION
    # ---------------------------------------------------------

    @staticmethod
    def _validate_numeric_column(
        dataframe: pd.DataFrame,
        column: str
    ) -> None:

        if column not in dataframe.columns:

            raise ValueError(
                f"Column '{column}' "
                f"does not exist in the dataset."
            )

        if not (
            pd.api.types.is_numeric_dtype(
                dataframe[column]
            )
        ):

            raise ValueError(
                f"Column '{column}' "
                f"is not numeric."
            )


    @staticmethod
    def _validate_group_by_column(
        dataframe: pd.DataFrame,
        group_by: str
    ) -> None:

        if group_by not in dataframe.columns:

            raise ValueError(
                f"Group-by column '{group_by}' "
                f"does not exist in the dataset."
            )

    def _argmax(self,dataframe: pd.DataFrame,column: str,subject: str):

        if dataframe.empty:
            return None

        self._validate_numeric_column(
            dataframe,
            column
        )

        max_value = dataframe[column].max()

        matching_rows = dataframe[
            dataframe[column] == max_value
        ]

        return matching_rows[
            [subject, column]
        ].drop_duplicates()


    def _argmin(self,dataframe: pd.DataFrame,column: str,subject: str):

        if dataframe.empty:
            return None

        self._validate_numeric_column(
            dataframe,
            column
        )

        min_value = dataframe[column].min()

        matching_rows = dataframe[
            dataframe[column] == min_value
        ]

        return matching_rows[
            [subject, column]
        ].drop_duplicates()        


    # ---------------------------------------------------------
    # GROUPED SUM
    # ---------------------------------------------------------

    def _grouped_sum(
        self,
        dataframe: pd.DataFrame,
        column: str,
        group_by: str
    ) -> pd.DataFrame:

        self._validate_numeric_column(
            dataframe,
            column
        )

        self._validate_group_by_column(
            dataframe,
            group_by
        )

        return (
            dataframe
            .groupby(group_by, dropna=False)[column]
            .sum()
            .reset_index(name=column)
        )


    # ---------------------------------------------------------
    # GROUPED AVERAGE
    # ---------------------------------------------------------

    def _grouped_average(
        self,
        dataframe: pd.DataFrame,
        column: str,
        group_by: str
    ) -> pd.DataFrame:

        self._validate_numeric_column(
            dataframe,
            column
        )

        self._validate_group_by_column(
            dataframe,
            group_by
        )

        return (
            dataframe
            .groupby(group_by, dropna=False)[column]
            .mean()
            .reset_index(name=column)
        )


    # ---------------------------------------------------------
    # GROUPED MINIMUM
    # ---------------------------------------------------------

    def _grouped_minimum(
        self,
        dataframe: pd.DataFrame,
        column: str,
        group_by: str
    ) -> pd.DataFrame:

        self._validate_numeric_column(
            dataframe,
            column
        )

        self._validate_group_by_column(
            dataframe,
            group_by
        )

        return (
            dataframe
            .groupby(group_by, dropna=False)[column]
            .min()
            .reset_index(name=column)
        )


    # ---------------------------------------------------------
    # GROUPED MAXIMUM
    # ---------------------------------------------------------

    def _grouped_maximum(
        self,
        dataframe: pd.DataFrame,
        column: str,
        group_by: str
    ) -> pd.DataFrame:

        self._validate_numeric_column(
            dataframe,
            column
        )

        self._validate_group_by_column(
            dataframe,
            group_by
        )

        return (
            dataframe
            .groupby(group_by, dropna=False)[column]
            .max()
            .reset_index(name=column)
        )

    # ---------------------------------------------------------
    # TOP N
    # ---------------------------------------------------------

    def _top_n(
        self,
        dataframe: pd.DataFrame,
        column: str,
        subject: str,
        limit: int | None
    ) -> pd.DataFrame:
        """
        Return the top N groups based on the total value
        of the requested numeric column.

        Example:

            Which are the top 3 regions by Revenue?

        Result:

            Region      Revenue
            North           842
            West            795
            South           731
        """

        if dataframe.empty:
            return pd.DataFrame(
                columns=[subject, column]
            )

        if subject is None:
            raise ValueError(
                "TOP_N operation requires a subject/group column."
            )

        if limit is None:
            raise ValueError(
                "TOP_N operation requires a limit."
            )

        if limit <= 0:
            raise ValueError(
                "TOP_N limit must be greater than zero."
            )

        # Validate measure column
        self._validate_numeric_column(
            dataframe,
            column
        )

        # Validate grouping column
        self._validate_group_by_column(
            dataframe,
            subject
        )

        # Aggregate the measure for each group
        result = (
            dataframe
            .groupby(
                subject,
                dropna=False
            )[column]
            .sum()
            .reset_index(name=column)
        )

        # Sort highest first
        result = result.sort_values(
            by=column,
            ascending=False
        )

        # Return top N
        return result.head(limit).reset_index(
            drop=True
        )


    # ---------------------------------------------------------
    # BOTTOM N
    # ---------------------------------------------------------

    def _bottom_n(
        self,
        dataframe: pd.DataFrame,
        column: str,
        subject: str,
        limit: int | None
    ) -> pd.DataFrame:
        """
        Return the bottom N groups based on the total value
        of the requested numeric column.

        Example:

            Which are the bottom 3 regions by Revenue?

        Result:

            Region      Revenue
            East            421
            Central         398
            Coastal         352
        """

        if dataframe.empty:
            return pd.DataFrame(
                columns=[subject, column]
            )

        if subject is None:
            raise ValueError(
                "BOTTOM_N operation requires a subject/group column."
            )

        if limit is None:
            raise ValueError(
                "BOTTOM_N operation requires a limit."
            )

        if limit <= 0:
            raise ValueError(
                "BOTTOM_N limit must be greater than zero."
            )

        # Validate measure column
        self._validate_numeric_column(
            dataframe,
            column
        )

        # Validate grouping column
        self._validate_group_by_column(
            dataframe,
            subject
        )

        # Aggregate the measure for each group
        result = (
            dataframe
            .groupby(
                subject,
                dropna=False
            )[column]
            .sum()
            .reset_index(name=column)
        )

        # Sort lowest first
        result = result.sort_values(
            by=column,
            ascending=True
        )

        # Return bottom N
        return result.head(limit).reset_index(
            drop=True
        )




