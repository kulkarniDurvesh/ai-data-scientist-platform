import pandas as pd

from core.date_parts import DATE_PART_OPERATORS, describe_date_part
from core.schema_inference import format_number, key_entity_name


AGGREGATION_WORDS = {
    "SUM": "total",
    "AVERAGE": "average",
    "MIN": "minimum",
    "MAX": "maximum",
}


class AnswerGenerator:
    """
    Converts raw query execution results into
    natural-language answers.

    This class is responsible only for presenting
    query execution results. It does not perform
    parsing, planning, or data analysis.
    """

    def generate(
        self,
        question: str,
        parsed_query,
        query_plan,
        result
    ) -> str:

        answer = self._generate_answer(
            question,
            parsed_query,
            query_plan,
            result
        )

        notes = getattr(parsed_query, "notes", None) or []

        return " ".join([answer, *notes])

    def _generate_answer(
        self,
        question: str,
        parsed_query,
        query_plan,
        result
    ) -> str:

        if parsed_query is None:
            raise ValueError(
                "Parsed query cannot be None."
            )

        if query_plan is None:
            raise ValueError(
                "Query plan cannot be None."
            )

        intent = parsed_query.intent

        # -------------------------------------------------
        # ARGMAX / ARGMIN
        # -------------------------------------------------

        if query_plan.operation == "ARGMAX":

            return self._generate_argmax_answer(
                query_plan,
                result
            )

        if query_plan.operation == "ARGMIN":

            return self._generate_argmin_answer(
                query_plan,
                result
            )

        # -------------------------------------------------
        # TOP N / BOTTOM N
        # -------------------------------------------------

        if query_plan.operation in ("TOP_N", "BOTTOM_N"):

            return self._generate_n_answer(
                query_plan,
                result
            )

        if query_plan.operation == "CHART":

            return self._generate_chart_answer(
                parsed_query,
                query_plan,
                result
            )
        # -------------------------------------------------
        # Standard operations
        # -------------------------------------------------

        if intent == "COUNT":

            return self._generate_count_answer(
                parsed_query.entity,
                parsed_query.filters,
                result
            )

        if intent == "AVERAGE":

            return self._generate_average_answer(
                parsed_query.entity,
                query_plan,
                result
            )

        if intent == "SUM":

            return self._generate_sum_answer(
                parsed_query.entity,
                query_plan,
                result
            )

        if intent == "MIN":

            return self._generate_min_answer(
                parsed_query.entity,
                query_plan,
                result
            )

        if intent == "MAX":

            return self._generate_max_answer(
                parsed_query.entity,
                query_plan,
                result
            )

        if intent == "FILTER":

            return self._generate_filter_answer(
                parsed_query.entity,
                parsed_query.filters,
                result
            )

        return (
            f"The query returned the following result: "
            f"{result}"
        )

    # =====================================================
    # COUNT
    # =====================================================

    def _generate_count_answer(
        self,
        entity,
        filters,
        result
    ) -> str:

        entity_name = self._get_entity_name(
            entity
        )

        filter_text = self._format_filters(
            filters
        )

        if filter_text:

            return (
                f"There are {result} "
                f"{entity_name} matching "
                f"{filter_text}."
            )

        return (
            f"There are {result} "
            f"{entity_name}."
        )

    # =====================================================
    # AVERAGE
    # =====================================================

    def _generate_average_answer(
        self,
        entity,
        query_plan,
        result
    ) -> str:

        column_name = self._format_column_name(
            entity
        )

        # -------------------------------------------------
        # GROUPED AVERAGE
        # -------------------------------------------------

        if query_plan.group_by is not None:

            if result is None or result.empty:

                return (
                    f"No values were found for "
                    f"{column_name}."
                )

            group_column = query_plan.group_by

            group_column_name = self._format_column_name(
                group_column
            )

            lines = []

            for _, row in result.iterrows():

                group_value = row[group_column]
                average_value = row[entity]

                lines.append(
                    f"{group_value}: "
                    f"{float(average_value):.2f}"
                )

            return (
                f"The average {column_name} "
                f"by {group_column_name} is:\n"
                + "\n".join(lines)
            )

        # -------------------------------------------------
        # NORMAL AVERAGE
        # -------------------------------------------------

        return (
            f"The average {column_name} "
            f"is {float(result):.2f}."
        )

    # =====================================================
    # SUM
    # =====================================================

    def _generate_sum_answer(self,entity,query_plan,result) -> str:

        column_name = self._format_column_name(
            entity
        )

        # -------------------------------------------------
        # GROUPED SUM
        # -------------------------------------------------

        if query_plan.group_by is not None:

            if result is None or result.empty:

                return (
                    f"No values were found for "
                    f"{column_name}."
                )

            group_column = query_plan.group_by

            group_column_name = self._format_column_name(
                group_column
            )

            lines = []

            for _, row in result.iterrows():

                group_value = row[group_column]
                sum_value = row[entity]

                lines.append(
                    f"{group_value}: "
                    f"{sum_value}"
                )

            return (
                f"The sum of {column_name} "
                f"by {group_column_name} is:\n"
                + "\n".join(lines)
            )



        return (
            f"The total {column_name} "
            f"is {float(result):.2f}."
        )

    # =====================================================
    # MIN
    # =====================================================

    def _generate_min_answer(self,entity,query_plan,result) -> str:

        column_name = self._format_column_name(
            entity
        )

        # -------------------------------------------------
        # GROUPED MINIMUM
        # -------------------------------------------------

        if query_plan.group_by is not None:

            if result is None or result.empty:

                return (
                    f"No values were found for "
                    f"{column_name}."
                )

            group_column = query_plan.group_by

            group_column_name = self._format_column_name(
                group_column
            )

            lines = []

            for _, row in result.iterrows():

                group_value = row[group_column]
                minimum_value = row[entity]

                lines.append(
                    f"{group_value}: "
                    f"{minimum_value}"
                )

            return (
                f"The minimum {column_name} "
                f"by {group_column_name} is:\n"
                + "\n".join(lines)
            )

        # -------------------------------------------------
        # NORMAL MINIMUM
        # -------------------------------------------------

        if result is None:

            return (
                f"No values were found for "
                f"{column_name}."
            )

        return (
            f"The minimum {column_name} "
            f"is {result}."
        )

    # =====================================================
    # MAX
    # =====================================================

    def _generate_max_answer(self,entity,query_plan,result) -> str:

        column_name = self._format_column_name(
            entity
        )

        # -------------------------------------------------
        # GROUPED MAXIMUM
        # -------------------------------------------------

        if query_plan.group_by is not None:

            if result is None or result.empty:

                return (
                    f"No values were found for "
                    f"{column_name}."
                )

            group_column = query_plan.group_by

            group_column_name = self._format_column_name(
                group_column
            )

            lines = []

            for _, row in result.iterrows():

                group_value = row[group_column]
                maximum_value = row[entity]

                lines.append(
                    f"{group_value}: "
                    f"{maximum_value}"
                )

            return (
                f"The maximum {column_name} "
                f"by {group_column_name} is:\n"
                + "\n".join(lines)
            )


        if result is None:

            return (
                f"No values were found for "
                f"{column_name}."
            )

        return (
            f"The maximum {column_name} "
            f"is {result}."
        )


    # =====================================================
    # TOP N / BOTTOM N
    # =====================================================

    def _generate_n_answer(
        self,
        query_plan,
        result
    ) -> str:

        if result is None or result.empty:

            return (
                "No matching records were found."
            )

        operation = query_plan.operation
        subject = query_plan.subject
        value_column = query_plan.column
        limit = query_plan.limit

        # -------------------------------------------------
        # Validate result columns
        # -------------------------------------------------

        if (
            subject not in result.columns
            or value_column not in result.columns
        ):

            return (
                "The query returned a result, "
                "but the requested entities could "
                "not be determined."
            )

        # -------------------------------------------------
        # Format names
        # -------------------------------------------------

        subject_name = self._get_entity_name(
            subject
        )

        value_name = self._format_column_name(
            value_column
        )

        # -------------------------------------------------
        # TOP / BOTTOM wording
        # -------------------------------------------------

        if operation == "TOP_N":

            position = "top"

        else:

            position = "bottom"

        # -------------------------------------------------
        # Build result lines
        # -------------------------------------------------

        lines = []

        for index, row in result.iterrows():

            subject_value = row[subject]
            measure_value = row[value_column]

            # Format numeric values with commas
            if isinstance(
                measure_value,
                (int, float)
            ):

                if float(measure_value).is_integer():

                    formatted_value = (
                        f"{int(measure_value):,}"
                    )

                else:

                    formatted_value = (
                        f"{measure_value:,.2f}"
                    )

            else:

                formatted_value = str(
                    measure_value
                )

            lines.append(
                f"{index + 1}. "
                f"{subject_value} — "
                f"{formatted_value}"
            )

        # -------------------------------------------------
        # Final answer
        # -------------------------------------------------

        return (
            f"The {position} {limit} "
            f"{subject_name} by "
            f"{value_name} are:\n"
            + "\n".join(lines)
        )


    # =====================================================
    # FILTER
    # =====================================================

    def _generate_filter_answer(
        self,
        entity,
        filters,
        result
    ) -> str:

        if result is None or result.empty:

            return (
                "No records matched the "
                "specified conditions."
            )

        row_count = len(result)

        if result.attrs.get("distinct_entity"):

            return self._generate_list_answer(
                entity,
                filters,
                row_count
            )

        entity_name = self._get_entity_name(
            entity
        )

        filter_text = self._format_filters(
            filters
        )

        if filter_text:

            return (
                f"Found {row_count} records "
                f"for {entity_name} matching "
                f"{filter_text}."
            )

        return (
            f"Found {row_count} records "
            f"for {entity_name}."
        )

    def _generate_list_answer(
        self,
        entity,
        filters,
        row_count
    ) -> str:

        entity_name = self._get_entity_name(entity)

        if row_count == 1:
            entity_name = self._singularize(entity_name)

        filter_text = self._format_filters(filters)

        if filter_text:
            return f"Found {row_count} {entity_name} where {filter_text}."

        return f"Found {row_count} {entity_name}."

    # =====================================================
    # CHART
    # =====================================================

    def _generate_chart_answer(
        self,
        parsed_query,
        query_plan,
        result
    ) -> str:

        if result is None or result.empty:

            return (
                "No records matched the "
                "specified conditions."
            )

        chart = (query_plan.chart_type or "bar").lower()
        group = query_plan.group_by
        aggregation = query_plan.aggregation

        if aggregation == "COUNT":
            measure = "records"
        elif aggregation == "COUNT_DISTINCT":
            measure = self._get_entity_name(query_plan.column)
        else:
            name = self._format_column_name(query_plan.column)
            word = AGGREGATION_WORDS.get(aggregation, aggregation.lower())
            # "Total Visits" already says how it is aggregated.
            measure = name if name.lower().startswith(word) else f"{word} {name}"

        filter_text = self._format_filters(parsed_query.filters)
        where = f" where {filter_text}" if filter_text else ""

        if group is None:
            return f"Here is a {chart} of {measure}{where}."

        text = (
            f"Here is a {chart} chart of {measure} by "
            f"{self._format_column_name(group)}{where}."
        )

        value_column = result.attrs.get("value_column")

        if (
            value_column
            and len(result) > 1
            and not pd.api.types.is_datetime64_any_dtype(result[group])
        ):
            top = result.iloc[0]
            text += (
                f" {len(result)} groups; {top[group]} is the largest "
                f"({format_number(top[value_column])}"
            )

            if aggregation in ("COUNT", "COUNT_DISTINCT"):
                text += f" of {format_number(result[value_column].sum())}"

            text += ")."

        return text

    # =====================================================
    # ARGMAX
    # =====================================================

    def _generate_argmax_answer(
        self,
        query_plan,
        result
    ) -> str:

        if result is None or result.empty:

            return (
                "No matching records were found."
            )

        subject = query_plan.subject
        value_column = query_plan.column

        if (
            subject not in result.columns
            or value_column not in result.columns
        ):

            return (
                "The maximum value was found, "
                "but the associated entity could "
                "not be determined."
            )

        max_value = result[
            value_column
        ].iloc[0]

        subjects = (
            result[subject]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        subject_name = self._get_entity_name(
            subject
        )

        value_name = self._format_column_name(
            value_column
        )

        if len(subjects) == 1:

            return (
                f"The {self._singularize(subject_name)} with "
                f"the highest {value_name} is "
                f"{subjects[0]}, with "
                f"{max_value}."
            )

        return (
            f"The highest {value_name} is "
            f"{max_value}, shared by "
            f"{len(subjects)} {subject_name}."
        )

    # =====================================================
    # ARGMIN
    # =====================================================

    def _generate_argmin_answer(
        self,
        query_plan,
        result
    ) -> str:

        if result is None or result.empty:

            return (
                "No matching records were found."
            )

        subject = query_plan.subject
        value_column = query_plan.column

        if (
            subject not in result.columns
            or value_column not in result.columns
        ):

            return (
                "The minimum value was found, "
                "but the associated entity could "
                "not be determined."
            )

        min_value = result[
            value_column
        ].iloc[0]

        subjects = (
            result[subject]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        subject_name = self._get_entity_name(
            subject
        )

        value_name = self._format_column_name(
            value_column
        )

        if len(subjects) == 1:

            return (
                f"The {self._singularize(subject_name)} with "
                f"the lowest {value_name} is "
                f"{subjects[0]}, with "
                f"{min_value}."
            )

        return (
            f"The lowest {value_name} is "
            f"{min_value}, shared by "
            f"{len(subjects)} {subject_name}."
        )

    # =====================================================
    # FILTER FORMATTING
    # =====================================================

    def _format_filters(
        self,
        filters
    ) -> str:

        if not filters:
            return ""

        parts = []

        # Calendar filters on one column read as one phrase:
        # "Visit Date is in March 2025".
        date_parts: dict[str, list] = {}

        for filter_item in filters:
            if filter_item.operator in DATE_PART_OPERATORS:
                date_parts.setdefault(filter_item.column, []).append(filter_item)

        for column_name, items in date_parts.items():
            items = sorted(items, key=lambda item: item.operator == "year")
            label = " ".join(
                describe_date_part(item.operator, item.value)
                for item in items
            )
            parts.append(
                f"{self._format_column_name(column_name)} is in {label}"
            )

        for filter_item in filters:

            if filter_item.operator in DATE_PART_OPERATORS:
                continue

            column = self._format_column_name(
                filter_item.column
            )

            operator = filter_item.operator
            value = filter_item.value

            if operator == "=":

                parts.append(
                    f"{column} is {value}"
                )

            elif operator == "!=":

                parts.append(
                    f"{column} is not {value}"
                )

            elif operator == ">":

                parts.append(
                    f"{column} is greater than {value}"
                )

            elif operator == "<":

                parts.append(
                    f"{column} is less than {value}"
                )

            elif operator == ">=":

                parts.append(
                    f"{column} is at least {value}"
                )

            elif operator == "<=":

                parts.append(
                    f"{column} is at most {value}"
                )

            else:

                parts.append(
                    f"{column} {operator} {value}"
                )

        return " and ".join(parts)

    # =====================================================
    # ENTITY NAME
    # =====================================================

    def _get_entity_name(
        self,
        entity
    ) -> str:

        if not entity:

            return "records"

        # Key columns name the entity they identify:
        # CustomerId -> customers, order_id -> orders
        base = key_entity_name(entity)

        if base is not None:

            return self._pluralize(base)

        name = self._format_column_name(
            entity
        ).lower()

        # Measures such as "total visits" are already plural.
        if name.endswith("s"):

            return name

        return self._pluralize(name)

    @staticmethod
    def _singularize(
        phrase: str
    ) -> str:

        if phrase.endswith("ies"):

            return phrase[:-3] + "y"

        if phrase.endswith(("ses", "xes", "zes", "ches", "shes")):

            return phrase[:-2]

        if phrase.endswith("s") and not phrase.endswith("ss"):

            return phrase[:-1]

        return phrase

    @staticmethod
    def _pluralize(
        phrase: str
    ) -> str:

        if phrase.endswith("y") and not phrase.endswith(
            ("ay", "ey", "oy", "uy")
        ):

            return phrase[:-1] + "ies"

        if phrase.endswith(("s", "x", "z", "ch", "sh")):

            return phrase + "es"

        return phrase + "s"

    # =====================================================
    # COLUMN NAME FORMATTING
    # =====================================================

    @staticmethod
    def _format_column_name(
        column
    ) -> str:

        if not column:

            return "records"

        result = ""

        for index, char in enumerate(column):

            if (
                index > 0
                and char.isupper()
                and not column[index - 1].isupper()
                and not column[index - 1].isspace()
            ):

                result += " "

            result += char

        return result





