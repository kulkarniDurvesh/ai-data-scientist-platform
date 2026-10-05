from dataclasses import dataclass, field
from typing import Any

from core.date_parts import DATE_PART_OPERATORS
from core.schema_inference import is_identifier


@dataclass
class QueryPlan:
    """
    Represents the execution plan generated from a ParsedQuery.

    The QueryPlan contains only dataset-related information
    required by the execution engine.

    Example:

        Query:
            How many customers are there in North region?

        Plan:
            operation = COUNT
            column = CustomerId
            filters = [
                {
                    "column": "Region",
                    "operator": "=",
                    "value": "North"
                }
            ]
    """

    operation: str

    column: str | None = None

    filters: list[dict[str, Any]] = field(
        default_factory=list
    )

    subject: str | None = None

    group_by: str | None = None

    aggregation: str | None = None

    limit: int | None = None

    chart_type: str | None = None

    time_grain: str | None = None

    notes: list[str] = field(
        default_factory=list
    )

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the QueryPlan into a dictionary.
        """

        return {
            "operation": self.operation,
            "column": self.column,
            "filters": self.filters,
            "subject": self.subject,
            "group_by": self.group_by,
            "aggregation": self.aggregation,
            "limit": self.limit,
            "chart_type": self.chart_type,
            "time_grain": self.time_grain,
            "notes": self.notes,
        }


class QueryPlanner:
    """
    Converts a ParsedQuery into a QueryPlan.

    The planner is responsible for mapping the parsed
    natural-language intent/entity/filter information
    into a structure that the Pandas execution engine
    can execute.
    """

    def __init__(self, dataframe, schema=None):

        self.df = dataframe

        self.schema = schema

        self.columns = list(
            dataframe.columns
        )

        self.column_lookup = {
            column.lower(): column
            for column in self.columns
        }

    # ---------------------------------------------------------
    # PUBLIC API
    # ---------------------------------------------------------

    def create_plan(
        self,
        parsed_query
    ) -> QueryPlan:
        """
        Convert ParsedQuery into QueryPlan.
        """

        if parsed_query is None:

            raise ValueError(
                "Parsed query cannot be None."
            )

        # -----------------------------------------------------
        # Determine operation
        # -----------------------------------------------------

        operation = parsed_query.intent

        # MAX + subject
        #
        # Example:
        # Which Region has the maximum Revenue?
        #
        # becomes:
        #
        # ARGMAX
        #
        if (
            parsed_query.intent == "MAX"
            and parsed_query.subject is not None
        ):

            operation = "ARGMAX"

        # MIN + subject
        #
        # Example:
        # Which Region has the minimum Revenue?
        #
        # becomes:
        #
        # ARGMIN
        #
        elif (
            parsed_query.intent == "MIN"
            and parsed_query.subject is not None
        ):

            operation = "ARGMIN"

        # -----------------------------------------------------
        # Resolve entity column
        # -----------------------------------------------------

        column = self._resolve_column(
            parsed_query.entity
        )

        # -----------------------------------------------------
        # Convert filters
        # -----------------------------------------------------

        filters = self._convert_filters(
            parsed_query.filters
        )

        # -----------------------------------------------------
        # Validate plan
        # -----------------------------------------------------

        self._validate_plan(
            operation,
            column,
            filters
        )

        aggregation = parsed_query.aggregation

        if operation == "CHART":
            aggregation = self._chart_aggregation(
                column,
                aggregation
            )

        # -----------------------------------------------------
        # Create QueryPlan
        # -----------------------------------------------------

        return QueryPlan(
            operation=operation,
            column=column,
            filters=filters,
            subject=parsed_query.subject,
            group_by=parsed_query.group_by,
            aggregation=aggregation,
            limit=parsed_query.limit,
            chart_type=parsed_query.chart_type,
            time_grain=parsed_query.time_grain,
            notes=list(parsed_query.notes),
        )

    def _chart_aggregation(
        self,
        column: str | None,
        requested: str | None
    ) -> str:
        """
        No column -> count rows; an entity or text column -> count
        distinct values; a numeric measure -> the requested
        aggregation, SUM by default.
        """

        if column is None:
            return "COUNT"

        if (
            is_identifier(column, self.schema)
            or self.df[column].dtype.kind not in "biufc"
        ):
            return "COUNT_DISTINCT"

        if requested in {"SUM", "AVERAGE", "MIN", "MAX"}:
            return requested

        return "SUM"

    # ---------------------------------------------------------
    # COLUMN RESOLUTION
    # ---------------------------------------------------------

    def _resolve_column(
        self,
        entity: str | None
    ) -> str | None:
        """
        Resolve a parsed entity to an actual dataframe column.
        """

        if entity is None:

            return None

        entity_lower = (
            entity
            .lower()
            .strip()
        )

        # -----------------------------------------------------
        # Exact column lookup
        # -----------------------------------------------------

        if entity_lower in self.column_lookup:

            return self.column_lookup[
                entity_lower
            ]

        # -----------------------------------------------------
        # Case-insensitive fallback
        # -----------------------------------------------------

        for column in self.columns:

            if column.lower() == entity_lower:

                return column

        raise ValueError(
            f"Column '{entity}' does not exist "
            f"in the dataset."
        )

    # ---------------------------------------------------------
    # FILTER CONVERSION
    # ---------------------------------------------------------

    def _convert_filters(
        self,
        filters
    ) -> list[dict[str, Any]]:
        """
        Convert parser filter objects into dictionaries
        understood by the execution engine.
        """

        result = []

        for filter_item in filters:

            column = self._resolve_filter_column(
                filter_item.column
            )

            self._validate_operator(
                filter_item.operator
            )

            result.append(
                {
                    "column": column,
                    "operator": filter_item.operator,
                    "value": filter_item.value
                }
            )

        return result

    # ---------------------------------------------------------
    # FILTER COLUMN RESOLUTION
    # ---------------------------------------------------------

    def _resolve_filter_column(
        self,
        column: str
    ) -> str:
        """
        Resolve filter column against dataframe columns.
        """

        if not column:

            raise ValueError(
                "Filter column cannot be empty."
            )

        column_lower = (
            column
            .lower()
            .strip()
        )

        if column_lower in self.column_lookup:

            return self.column_lookup[
                column_lower
            ]

        raise ValueError(
            f"Filter column '{column}' does not exist "
            f"in the dataset."
        )

    # ---------------------------------------------------------
    # OPERATOR VALIDATION
    # ---------------------------------------------------------

    @staticmethod
    def _validate_operator(
        operator: str
    ) -> None:
        """
        Validate supported filter operators.
        """

        allowed_operators = {
            "=",
            ">",
            "<",
            ">=",
            "<=",
            "!="
        } | DATE_PART_OPERATORS

        if operator not in allowed_operators:

            raise ValueError(
                f"Unsupported filter operator: "
                f"'{operator}'"
            )

    # ---------------------------------------------------------
    # PLAN VALIDATION
    # ---------------------------------------------------------

    def _validate_plan(
        self,
        operation: str,
        column: str | None,
        filters: list[dict[str, Any]]
    ) -> None:
        """
        Validate that the operation is supported and
        has the required information.
        """

        supported_operations = {
            "COUNT",
            "SUM",
            "AVERAGE",
            "MIN",
            "MAX",
            "ARGMAX",
            "ARGMIN",
            "FILTER",
            "COMPARE",
            "TOP_N",
            "BOTTOM_N",
            "CHART"
        }

        # -----------------------------------------------------
        # Operation validation
        # -----------------------------------------------------

        if operation not in supported_operations:

            raise ValueError(
                f"Unsupported query operation: "
                f"'{operation}'"
            )

        # -----------------------------------------------------
        # Column validation
        # -----------------------------------------------------

        if (
            operation not in ("FILTER", "CHART")
            and column is None
        ):

            raise ValueError(
                f"Operation '{operation}' requires "
                f"a dataset column."
            )


            