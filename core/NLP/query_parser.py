
import difflib
import re
from typing import Any

from core.date_parts import (
    as_datetime,
    choose_date_column,
    date_part_mask,
    describe_date_part,
    detect_date_parts,
)
from core.schema_inference import is_identifier, key_entity_name, name_tokens, same_word

from .query_schema import ParsedQuery, QueryFilter
from .query_vocabulary import (
    CHART_TYPE_PATTERNS,
    CHART_WORDS,
    FALLBACK_LIST_WORDS,
    INTENT_KEYWORDS,
    TIME_GRAIN_WORDS,
)


# ID-like token in a question: "MR01", "cust-12", "DOC0005".
CODE_TOKEN = re.compile(r"\b([a-z]{2,})[-_]?(\d+)\b")
# ID-like value in a column: "MR001", "CUST-0012".
CODE_VALUE = re.compile(r"^([A-Za-z]+)[-_ ]?(\d+)$")

# Phrases that introduce the grouping of a chart.
GROUP_PHRASE = re.compile(
    r"\b(?:grouped by|split by|broken down by|according to|based on|"
    r"for each|for every|by|per|across)\s+"
    r"(?:their\s+|the\s+|each\s+|its\s+)?"
    r"([a-z0-9_ ]+?)"
    r"(?=\s+(?:in|for|where|among|during|of|with|and|from|as|on|wise)\b|\s*[?,.]|\s*$)"
)
# A word with a lower-to-upper case change: "doctorsId", "NorthTerritory".
GLUED_WORD = re.compile(r"\b[A-Za-z]*[a-z][A-Z][A-Za-z]*\b")

WISE_PHRASE = re.compile(r"\b([a-z0-9_]+)[\s-]?wise\b")


def code_key(value: Any) -> tuple[str, int] | None:
    """("mr", 1) for "MR001", "MR01" or "mr-1"; None if not ID-like."""

    match = CODE_VALUE.match(str(value).strip())

    if match is None:
        return None

    return match.group(1).lower(), int(match.group(2))


class QueryParser:
    """
    Converts a natural-language question into a ParsedQuery.

    The parser uses dataset metadata/profile to resolve:
        - columns
        - column aliases
        - categorical values
        - numeric values
        - intents
        - operators
    """

    # Categorical value matching is limited to columns with at most
    # this many distinct values (free text / IDs are skipped).
    MAX_FILTER_VALUES = 1000

    def __init__(self, dataframe, schema=None):
        self.df = dataframe
        self.schema = schema

        self.columns = list(dataframe.columns)

        self.filterable_columns = self._build_filterable_columns()

        # Lowercase lookup for case-insensitive column matching.
        self.column_lookup = {
            column.lower(): column
            for column in self.columns
        }

        self.column_aliases = self._build_column_aliases()

    # ---------------------------------------------------------
    # PUBLIC API
    # ---------------------------------------------------------

    def measure_in(self, question: str) -> str | None:
        """The numeric column a question is about ("why did revenue drop")."""
        return self._find_numeric_column_from_context(self._split_glued_words(question))

    def entity_in(self, question: str) -> str | None:
        """The key column whose entities a question counts ("why did visits drop")."""
        return self._infer_count_entity(self._split_glued_words(question).lower())

    def values_in(self, question: str) -> list[QueryFilter]:
        """Filters for category values the question names exactly ("... in West")."""
        return self._detect_categorical_filters(self._split_glued_words(question))

    def parse(self, question: str) -> ParsedQuery:

        if not question or not question.strip():
            raise ValueError("Query cannot be empty.")

        question = self._split_glued_words(question.strip())

        intent = self._detect_intent(question)

        chart_type = None
        time_grain = None
        subject = None

        if intent == "CHART":
            chart_type = self._detect_chart_type(question)
            group_by, time_grain = self._detect_chart_group_by(
                question, chart_type
            )
            entity = self._detect_chart_measure(question, group_by)
        else:
            entity = self._detect_entity(
                question,
                intent
            )

            ranking_subject = self._detect_ranking_subject(question)

            subject = ranking_subject or self._detect_subject(question)

            if intent in ("TOP_N", "BOTTOM_N"):
                group_by = None
            else:
                group_by = self._detect_group_by(question)

        filters = self._detect_filters(
            question,
            entity
        )

        code_filters, unresolved = self._detect_code_filters(
            question,
            filters
        )

        unresolved += self._unknown_category_values(question, filters + code_filters)

        date_filters, notes = self._detect_date_filters(question)

        filters = self._remove_duplicate_filters(
            filters + code_filters + date_filters
        )

        aggregation = self._detect_aggregation(
            question,
            intent
        )

        limit = self._detect_limit(question)

        # A list request that names no column and no value would
        # return every row; say so instead.
        if intent == "FILTER" and entity is None and not filters:
            raise ValueError(
                "I couldn't match any column or value in that question. "
                "Name a column or a value as it appears in the data."
            )

        return ParsedQuery(
            intent=intent,
            entity=entity,
            filters=filters,
            subject=subject,
            group_by=group_by,
            aggregation=aggregation,
            limit=limit,
            chart_type=chart_type,
            time_grain=time_grain,
            notes=notes,
            unresolved=unresolved,
        )

    # ---------------------------------------------------------
    # INTENT
    # ---------------------------------------------------------

    def _detect_intent(self,question: str) -> str:

        text = question.lower()

        # -----------------------------------------------------
        # TOP N / BOTTOM N
        #
        # Ranking intent has priority over aggregation words
        # such as "average", "sum", "total", etc.
        # -----------------------------------------------------

        if re.search(
            r"\btop\s+\d+\b",
            text
        ):
            return "TOP_N"

        if re.search(
            r"\bbottom\s+\d+\b",
            text
        ):
            return "BOTTOM_N"

        # "pie chart of doctors by specialty"
        if self._detect_chart_type(question):
            return "CHART"

        # -----------------------------------------------------
        # Generic intent detection
        # -----------------------------------------------------

        matches = []

        for intent, keywords in INTENT_KEYWORDS.items():

            for keyword in keywords:

                if keyword.lower() in text:

                    matches.append(
                        (
                            len(keyword),
                            intent
                        )
                    )

        if not matches:

            # "Which doctors are in Pune North?" asks for a list.
            if any(
                re.search(rf"\b{re.escape(word)}\b", text)
                for word in FALLBACK_LIST_WORDS
            ):
                return "FILTER"

            raise ValueError(
                f"Unable to determine query intent from: "
                f"'{question}'"
            )

        matches.sort(reverse=True)

        return matches[0][1]
    # ---------------------------------------------------------
    # ENTITY
    # ---------------------------------------------------------

    def _detect_entity(
        self,
        question: str,
        intent: str
    ) -> str | None:

        text = question.lower()

        # -----------------------------------------------------
        # For COUNT, first try to identify the entity being
        # counted from ID columns.
        # -----------------------------------------------------

        # COUNT and list questions are about an entity ("doctors"),
        # which takes priority over other columns named in the
        # question ("... in Pune North territory").
        if intent in ("COUNT", "FILTER"):

            count_entity = self._infer_count_entity(text)

            if count_entity:
                return count_entity

        # Aggregations and rankings operate on a numeric column, so
        # when several columns are mentioned ("average Sales by
        # Region") the numeric one is the entity.
        prefer_numeric = intent in {
            "SUM", "AVERAGE", "MIN", "MAX", "TOP_N", "BOTTOM_N"
        }

        # -----------------------------------------------------
        # Exact column names
        # -----------------------------------------------------

        exact_matches = []

        for column in self.columns:

            column_lower = column.lower()

            if re.search(
                rf"\b{re.escape(column_lower)}\b",
                text
            ):

                exact_matches.append(
                    (
                        prefer_numeric and self._is_numeric_column(column),
                        len(column_lower),
                        column
                    )
                )

        if exact_matches:

            exact_matches.sort(
                reverse=True
            )

            return exact_matches[0][-1]

        # -----------------------------------------------------
        # Column aliases
        # -----------------------------------------------------

        alias_matches = []

        for alias, column in self.column_aliases.items():

            if re.search(
                rf"\b{re.escape(alias)}\b",
                text
            ):

                alias_matches.append(
                    (
                        prefer_numeric and self._is_numeric_column(column),
                        len(alias),
                        column
                    )
                )

        if alias_matches:

            alias_matches.sort(
                reverse=True
            )

            return alias_matches[0][-1]

        return None

    # ---------------------------------------------------------
    # COUNT ENTITY
    # ---------------------------------------------------------

    def _infer_count_entity(
        self,
        text: str
    ) -> str | None:

        candidates = []

        for column in self.columns:

            for singular in self._entity_names(column):

                plural = self._pluralize(
                    singular
                )

                if (
                    re.search(
                        rf"\b{re.escape(singular)}\b",
                        text
                    )
                    or
                    re.search(
                        rf"\b{re.escape(plural)}\b",
                        text
                    )
                ):

                    candidates.append(
                        (
                            len(singular),
                            column
                        )
                    )

        if candidates:

            candidates.sort(
                reverse=True
            )

            return candidates[0][1]

        return None

    # ---------------------------------------------------------
    # FILTERS
    # ---------------------------------------------------------

    def _detect_filters(
        self,
        question: str,
        entity: str | None
    ) -> list[QueryFilter]:

        filters = []

        text = question.lower()

        # -----------------------------------------------------
        # Numeric filters
        # -----------------------------------------------------
        numeric_filter_patterns = [

            (
                r"(>=)\s*(-?\d+(?:\.\d+)?)",
                ">="
            ),

            (
                r"(<=)\s*(-?\d+(?:\.\d+)?)",
                "<="
            ),

            (
                r"(>)\s*(-?\d+(?:\.\d+)?)",
                ">"
            ),

            (
                r"(<)\s*(-?\d+(?:\.\d+)?)",
                "<"
            ),

            (
                r"(=)\s*(-?\d+(?:\.\d+)?)",
                "="
            ),

            (
                r"(greater than or equal to|at least)\s+(-?\d+(?:\.\d+)?)",
                ">="
            ),

            (
                r"(less than or equal to|at most)\s+(-?\d+(?:\.\d+)?)",
                "<="
            ),

            (
                r"(greater than|more than|above|over)\s+(-?\d+(?:\.\d+)?)",
                ">"
            ),

            (
                r"(less than|below|under)\s+(-?\d+(?:\.\d+)?)",
                "<"
            ),

            (
                r"(equal to|equals)\s+(-?\d+(?:\.\d+)?)",
                "="
            )
        ]

        for pattern, operator in numeric_filter_patterns:
            for match in re.finditer(pattern, text):

                value = self._convert_number(match.group(2))

                column = self._find_numeric_column_near_filter(
                    question,
                    match.start()
                )

                if column is None:
                    column = self._find_numeric_column_from_context(question)

                if column:
                    filters.append(
                        QueryFilter(
                            column=column,
                            operator=operator,
                            value=value
                        )
                    )

        # -----------------------------------------------------
        # Categorical filters
        # -----------------------------------------------------

        filters.extend(
            self._detect_categorical_filters(
                question
            )
        )

        return self._remove_duplicate_filters(
            filters
        )

    # ---------------------------------------------------------
    # CATEGORICAL VALUES
    # ---------------------------------------------------------

    def _detect_categorical_filters(
        self,
        question: str
    ) -> list[QueryFilter]:

        text = question.lower()

        filters = []

        for column in self.filterable_columns:

            try:

                values = (
                    self.df[column]
                    .dropna()
                    .unique()
                )

            except Exception:

                continue

            for value in values:

                value_text = str(value).strip()

                if not value_text:
                    continue

                # Ignore very long free-text values.
                if len(value_text) > 50:
                    continue

                pattern = (
                    rf"\b{re.escape(value_text.lower())}\b"
                )

                if re.search(
                    pattern,
                    text
                ):

                    filters.append(
                        QueryFilter(
                            column=column,
                            operator="=",
                            value=self._restore_original_value(
                                column,
                                value_text
                            )
                        )
                    )

        return self._one_column_per_value(question, filters)

    # Words between a preposition and a column name that are not values:
    # "in each region", "for the same category".
    NOT_VALUES = {
        "each", "every", "all", "any", "which", "what", "that", "this", "the", "a", "an",
        "my", "our", "their", "its", "same", "one", "per", "whole", "entire", "every",
    }

    def _unknown_category_values(self, question: str, filters: list[QueryFilter]) -> list[str]:
        """
        "customers in Atlantis region": a value placed before a category
        column's name that the column doesn't have. Ignoring it would
        silently answer a different question, so it is reported instead.
        """

        text = question.lower()
        filtered = {f.column for f in filters}
        unknown = []

        for column in self.filterable_columns:
            if column in filtered or (self.schema is not None and self.schema.role_of(str(column)) == "identifier"):
                continue
            tokens = [t for t in self._column_name_words(column) if len(t) >= 3]
            if not tokens:
                continue
            name = re.escape(tokens[-1])
            for match in re.finditer(rf"\b(?:in|from|of|for|at|with)\s+((?:[\w\-]+\s+){{1,3}}?){name}s?\b", text):
                phrase = match.group(1).strip()
                words = phrase.split()
                if not words or any(w in self.NOT_VALUES or w in self.CONNECTORS for w in words):
                    continue
                if any(same_word(w, t) for w in words for t in self._all_name_words()):
                    continue
                values = {str(v).strip().lower() for v in self.df[column].dropna().unique()}
                if phrase in values or any(word in values for word in words):
                    continue
                original = question[match.start(1): match.start(1) + len(phrase)]
                if original not in unknown:
                    unknown.append(original)
        return unknown

    CONNECTORS = {"by", "per", "and", "or", "vs", "versus", "to", "across", "over", "than", "with", "without", "on", "as"}

    def _all_name_words(self) -> set[str]:
        if not hasattr(self, "_name_words"):
            self._name_words = {t for c in self.columns for t in self._column_name_words(c) if len(t) >= 3}
        return self._name_words

    def _column_name_words(self, column) -> list[str]:
        from core.schema_inference import name_tokens

        return name_tokens(column)

    def _one_column_per_value(
        self,
        question: str,
        filters: list[QueryFilter]
    ) -> list[QueryFilter]:
        """
        A value found in several columns ("Pune North" in a doctor's
        and an MR's territory) filters only the column the question
        points at ("doctors in Pune North"), else the first one.
        """

        words = [
            word
            for word in re.findall(r"[a-z]+", question.lower())
            if len(word) >= 3
        ]

        by_value: dict[str, list[QueryFilter]] = {}

        for item in filters:
            by_value.setdefault(str(item.value).lower(), []).append(item)

        result = []

        for items in by_value.values():

            if len(items) == 1:
                result.extend(items)
                continue

            value_words = set(re.findall(r"[a-z]+", str(items[0].value).lower()))

            def mentions(item: QueryFilter) -> int:
                return sum(
                    any(same_word(word, token) for word in words if word not in value_words)
                    for token in name_tokens(item.column)
                    if len(token) >= 3
                )

            result.append(max(items, key=mentions))

        return result

    # ---------------------------------------------------------
    # NUMERIC COLUMN DETECTION
    # ---------------------------------------------------------

    def _find_numeric_column_near_filter(
        self,
        question: str,
        filter_position: int
    ) -> str | None:

        text_before_filter = (
            question[:filter_position].lower()
        )

        candidates = []

        for column in self.columns:

            if not self._is_numeric_column(column):
                continue

            column_lower = column.lower()

            # -------------------------------------------------
            # Check actual column name.
            # Example:
            #
            # "TotalSales more than 20"
            # -------------------------------------------------

            position = text_before_filter.rfind(
                column_lower
            )

            if position >= 0:

                candidates.append(
                    (
                        filter_position - position,
                        column
                    )
                )

            # -------------------------------------------------
            # Check semantic aliases.
            # Example:
            #
            # "sales more than 20"
            #
            # sales → TotalSales
            # -------------------------------------------------

            for alias, alias_column in self.column_aliases.items():

                if alias_column != column:
                    continue

                alias_pattern = (
                    rf"\b{re.escape(alias)}\b"
                )

                matches = list(
                    re.finditer(
                        alias_pattern,
                        text_before_filter
                    )
                )

                for alias_match in matches:

                    candidates.append(
                        (
                            filter_position
                            - alias_match.start(),
                            column
                        )
                    )

        if candidates:

            candidates.sort(
                key=lambda item: item[0]
            )

            return candidates[0][1]

        return None

    def _find_numeric_column_from_context(
        self,
        question: str
    ) -> str | None:

        text = question.lower()

        candidates = []

        # -----------------------------------------------------
        # First check semantic aliases.
        #
        # Example:
        #
        # "more than 20 sales"
        #
        # sales → TotalSales
        # -----------------------------------------------------

        for alias, column in self.column_aliases.items():

            if not self._is_numeric_column(column):
                continue

            pattern = (
                rf"\b{re.escape(alias)}\b"
            )

            match = re.search(
                pattern,
                text
            )

            if match:

                candidates.append(
                    (
                        len(alias),
                        match.start(),
                        column
                    )
                )

        if candidates:

            # Prefer the longest semantic alias.
            candidates.sort(
                key=lambda item: (
                    -item[0],
                    item[1]
                )
            )

            return candidates[0][2]

        # -----------------------------------------------------
        # Check actual column names.
        # -----------------------------------------------------

        for column in self.columns:

            if not self._is_numeric_column(column):
                continue

            normalized = column.lower()

            if normalized in text:
                return column

        # -----------------------------------------------------
        # Inspect individual words from numeric columns.
        # -----------------------------------------------------

        candidates = []

        for column in self.columns:

            if not self._is_numeric_column(column):
                continue

            normalized = column.lower()

            words = re.findall(
                r"[a-z]+",
                normalized
            )

            for word in words:

                if len(word) < 4:
                    continue

                if re.search(
                    rf"\b{re.escape(word)}\b",
                    text
                ):

                    candidates.append(
                        (
                            len(word),
                            column
                        )
                    )

        if candidates:

            candidates.sort(
                key=lambda item: item[0],
                reverse=True
            )

            return candidates[0][1]

        return None

    # ---------------------------------------------------------
    # COLUMN ALIASES
    # ---------------------------------------------------------

    def _build_column_aliases(
        self
    ) -> dict[str, str]:

        aliases = {}

        for column in self.columns:

            column_lower = column.lower()

            # -------------------------------------------------
            # Original column name
            # -------------------------------------------------

            normalized = (
                column_lower
                .replace("_", " ")
            )

            aliases[normalized] = column

            # -------------------------------------------------
            # CamelCase → words
            #
            # TotalSales → total sales
            # CustomerId → customer id
            # -------------------------------------------------

            spaced = re.sub(
                r"(?<!^)(?=[A-Z])",
                " ",
                column
            ).lower()

            aliases[spaced] = column

            # -------------------------------------------------
            # ID columns
            #
            # CustomerId → customer
            # CustomerId → customers
            # -------------------------------------------------

            for base in self._entity_names(column):

                aliases[
                    base
                ] = column

                aliases[
                    self._pluralize(
                        base
                    )
                ] = column

            # -------------------------------------------------
            # Total measure columns
            #
            # TotalSales → sales
            # TotalOrders → orders
            #
            # This allows:
            #
            # "more than 20 sales"
            #
            # to resolve to:
            #
            # TotalSales > 20
            # -------------------------------------------------

            if column_lower.startswith("total"):

                base = column[5:].strip(
                    "_ "
                )

                if base:

                    base_lower = base.lower()

                    aliases[
                        base_lower
                    ] = column

                    aliases[
                        self._pluralize(
                            base_lower
                        )
                    ] = column

            # -------------------------------------------------
            # Date columns
            # -------------------------------------------------

            if column_lower.endswith("date"):

                base = column[:-4].strip(
                    "_ "
                )

                if base:

                    aliases[
                        f"{base.lower()} date"
                    ] = column

        return aliases

    def _build_filterable_columns(self) -> list[str]:
        """
        Non-numeric columns whose values can appear in a question
        as filters ("... in North region"). Uses the shared schema
        when available so free text is never scanned.
        """

        result = []

        for column in self.columns:

            if self._is_numeric_column(column):
                continue

            if self.schema is not None:

                role = self.schema.role_of(str(column))

                if role not in {"dimension", "binary", "identifier"}:
                    continue

                if (
                    self.schema.columns[str(column)].n_unique
                    > self.MAX_FILTER_VALUES
                ):
                    continue

            result.append(column)

        return result

    # ---------------------------------------------------------
    # DATA TYPE HELPERS
    # ---------------------------------------------------------

    def _is_numeric_column(
        self,
        column: str
    ) -> bool:

        return (
            self.df[column].dtype.kind
            in "biufc"
        )

    def _entity_names(
        self,
        column: str
    ) -> list[str]:
        """
        Words that name the entity of a key column, decided by the
        identifier role (not by the name ending in "id"):

            DoctorId   -> ["doctor"]
            SalesRepId -> ["sales rep", "salesrep"]
            AmountPaid -> []
        """

        if not is_identifier(column, self.schema):
            return []

        base = key_entity_name(column)

        if base is None:
            return []

        return list(dict.fromkeys([base, base.replace(" ", "")]))

    def _is_categorical_column(
        self,
        column: str
    ) -> bool:

        return not self._is_numeric_column(
            column
        )

    # ---------------------------------------------------------
    # UTILITIES
    # ---------------------------------------------------------

    @staticmethod
    def _convert_number(
        value: str
    ) -> Any:

        if "." in value:
            return float(value)

        return int(value)

    @staticmethod
    def _pluralize(
        word: str
    ) -> str:

        if word.endswith("y"):

            return (
                word[:-1]
                + "ies"
            )

        if word.endswith(
            ("s", "x", "z", "ch", "sh")
        ):

            return word + "es"

        return word + "s"

    def _restore_original_value(
        self,
        column: str,
        value: str
    ) -> Any:

        values = (
            self.df[column]
            .dropna()
            .unique()
        )

        for original in values:

            if (
                str(original).lower()
                == value.lower()
            ):

                return original

        return value

    @staticmethod
    def _remove_duplicate_filters(
        filters: list[QueryFilter]
    ) -> list[QueryFilter]:

        result = []

        seen = set()

        for filter_item in filters:

            key = (
                filter_item.column,
                filter_item.operator,
                str(
                    filter_item.value
                ).lower()
            )

            if key not in seen:

                seen.add(key)

                result.append(
                    filter_item
                )

        return result


    def _detect_subject(self, question: str) -> str | None:
        """
        Detects the entity being requested by the question.

        The phrase after "which" is resolved against the dataset's
        own column names and aliases, so no entity is hardcoded.

        Examples:

            Which customer has the highest Sales?
                -> CustomerId   (alias built from the ID column)

            Which region has the highest Sales?
                -> Region
        """

        question_lower = question.lower()

        match = re.search(
            r"\bwhich\s+([a-z0-9_ ]+?)\s+"
            r"(?:has|have|had|is|are|was|were|with|shows?|got|gets)\b",
            question_lower
        )

        if not match:
            return None

        phrase = match.group(1).strip()

        resolved = self._resolve_column_reference(phrase)

        if resolved:
            return resolved

        # Fall back to the last word: "which sales region has ..."
        words = phrase.split()

        if len(words) > 1:
            return self._resolve_column_reference(words[-1])

        return None

    # ---------------------------------------------------------
# GROUP BY
# ---------------------------------------------------------

    def _detect_group_by(self,question: str) -> str | None:
        """
        Detects the grouping column for comparison queries.

        Examples:

            What is the average TotalSales by Region?
                -> Region

            Show average Revenue by region.
                -> Region

            Compare TotalSales across regions.
                -> Region
        """

        text = question.lower()

        # -----------------------------------------------------
        # Explicit "by <column>" pattern
        # -----------------------------------------------------

        by_match = re.search(
            r"\bby\s+([a-zA-Z0-9_ ]+?)(?=\s+where\b|\?|$|,|\.)",
            text
        )

        if by_match:

            candidate = by_match.group(1).strip()

            # Try exact column / alias resolution
            resolved = self._resolve_column_reference(
                candidate
            )

            if resolved:
                return resolved

        # -----------------------------------------------------
        # "across <column>" pattern
        # -----------------------------------------------------

        across_match = re.search(
            r"\bacross\s+([a-zA-Z0-9_ ]+?)(?:\?|$|,|\.)",
            text
        )

        if across_match:

            candidate = across_match.group(1).strip()

            resolved = self._resolve_column_reference(
                candidate
            )

            if resolved:
                return resolved

        # -----------------------------------------------------
        # Generic semantic detection
        # -----------------------------------------------------

        comparison_words = [
            "compare",
            "comparison",
            "among",
            "between",
            "grouped",
            "per"
        ]

        if any(
            word in text
            for word in comparison_words
        ):

            # Look for categorical columns
            # mentioned in the question.

            candidates = []

            for alias, column in self.column_aliases.items():

                if self._is_numeric_column(column):
                    continue

                pattern = (
                    rf"\b{re.escape(alias)}\b"
                )

                match = re.search(
                    pattern,
                    text
                )

                if match:

                    candidates.append(
                        (
                            len(alias),
                            match.start(),
                            column
                        )
                    )

            if candidates:

                candidates.sort(
                    key=lambda item: (
                        -item[0],
                        item[1]
                    )
                )

                return candidates[0][2]

        return None

        # ---------------------------------------------------------
        # COLUMN REFERENCE RESOLUTION
        # ---------------------------------------------------------

    def _resolve_column_reference(self, reference: str) -> str | None:
        reference = self._normalize_reference(reference)

        for column in self.columns:
            if self._normalize_reference(column) == reference:
                return column

        for column in self.columns:
            normalized_column = column.lower().replace("_", " ")
            normalized_column = self._normalize_reference(
                normalized_column
            )

            if normalized_column == reference:
                return column

        if reference in self.column_aliases:
            return self.column_aliases[reference]

        candidates = []

        for alias, column in self.column_aliases.items():
            normalized_alias = self._normalize_reference(alias)

            if re.search(
                rf"\b{re.escape(normalized_alias)}\b",
                reference
            ):
                candidates.append(
                    (len(normalized_alias), column)
                )

        if candidates:
            candidates.sort(reverse=True)
            return candidates[0][1]

        # Spelling variants: "speciality" -> Specialty
        normalized = {
            self._normalize_reference(alias): column
            for alias, column in self.column_aliases.items()
        }

        close = difflib.get_close_matches(
            reference, list(normalized), n=1, cutoff=0.85
        )

        if close:
            return normalized[close[0]]

        return None

    def _detect_aggregation(self,question: str,intent: str) -> str | None:

        text = question.lower()

        aggregation_keywords = {
            "AVERAGE": [
                "average",
                "avg",
                "mean"
            ],

            "SUM": [
                "sum",
                "total"
            ],

            "COUNT": [
                "count",
                "number of",
                "how many"
            ],

            "MIN": [
                "minimum",
                "min",
                "lowest",
                "smallest"
            ],

            "MAX": [
                "maximum",
                "max",
                "highest",
                "largest"
            ]
        }

        # -----------------------------------------------------
        # TOP_N / BOTTOM_N
        #
        # Ranking already determines the operation.
        # Do not allow words inside column names such as
        # "TotalSales" to become SUM.
        # -----------------------------------------------------

        if intent in ("TOP_N", "BOTTOM_N"):

            return None

        matches = []

        for aggregation, keywords in aggregation_keywords.items():

            for keyword in keywords:

                if re.search(
                    rf"\b{re.escape(keyword)}\b",
                    text
                ):

                    matches.append(
                        (
                            len(keyword),
                            aggregation
                        )
                    )

        if matches:

            matches.sort(
                reverse=True
            )

            return matches[0][1]

        # -----------------------------------------------------
        # Preserve existing intent when appropriate
        # -----------------------------------------------------

        if intent in {
            "COUNT",
            "SUM",
            "AVERAGE",
            "MIN",
            "MAX"
        }:

            return intent

        return None

    def _detect_limit(self, question: str) -> int | None:
        text = question.lower()

        match = re.search(
            r"\b(?:top|bottom)\s+(\d+)\b",
            text
        )

        if match:
            return int(match.group(1))

        return None


    def _normalize_reference(self, value: str) -> str:
        value = value.strip().lower()
        value = re.sub(r"\s+", " ", value)

        if value.endswith("ies"):
            value = value[:-3] + "y"
        elif value.endswith("s"):
            value = value[:-1]

        return value

    def _detect_ranking_subject(self, question: str) -> str | None:
        text = question.lower()

        match = re.search(
            r"\b(?:top|bottom)\s+\d+\s+(.+?)\s+\bby\b",
            text
        )

        if not match:
            return None

        candidate = match.group(1).strip()

        return self._resolve_column_reference(candidate)


    # ---------------------------------------------------------
    # CHARTS
    # ---------------------------------------------------------

    @staticmethod
    def _detect_chart_type(question: str) -> str | None:
        """
        Chart type requested in the question, or None when no chart
        is asked for. "pie", "histogram" and "scatter" are enough on
        their own; other types need a word such as "chart" or "plot".
        """

        text = question.lower()

        asks_for_chart = re.search(CHART_WORDS, text) is not None

        for chart_type, pattern, standalone in CHART_TYPE_PATTERNS:
            if re.search(pattern, text) and (standalone or asks_for_chart):
                return chart_type

        return "BAR" if asks_for_chart else None

    def _detect_chart_group_by(
        self,
        question: str,
        chart_type: str | None,
    ) -> tuple[str | None, str | None]:
        """
        Column the chart is split by, plus a time grain when the
        question groups by a calendar unit ("visits by month").
        """

        text = question.lower()

        phrases = [
            match.group(1).strip()
            for match in GROUP_PHRASE.finditer(text)
        ] + [
            match.group(1)
            for match in WISE_PHRASE.finditer(text)
        ]

        for phrase in phrases:

            column = self._resolve_column_reference(phrase)

            if column is not None:
                return column, None

            words = phrase.split()
            unit = words[-1] if words else ""
            grain = TIME_GRAIN_WORDS.get(unit) or TIME_GRAIN_WORDS.get(unit.rstrip("s"))

            if grain:
                date_column = choose_date_column(question, self.df, self.schema)
                if date_column is not None:
                    return date_column, grain

        column = self._detect_group_by(question)

        if column is not None:
            return column, None

        if chart_type in ("HISTOGRAM", "BOX", "SCATTER"):
            return None, None

        # A trend without a grouping runs over time.
        if chart_type == "LINE":
            date_column = choose_date_column(question, self.df, self.schema)
            if date_column is not None:
                return date_column, None

        example = ""
        if self.schema is not None and self.schema.groupable:
            example = f" For example: 'by {self.schema.groupable[0]}'."

        raise ValueError(
            "Tell me what to split the chart by." + example
        )

    def _detect_chart_measure(
        self,
        question: str,
        group_by: str | None,
    ) -> str | None:
        """
        What the chart measures: an entity to count ("doctors") or a
        numeric column to aggregate ("revenue"). None counts rows.
        """

        text = question.lower()

        # "chart of visits by competitor": the entity before the
        # grouping phrase is what gets counted.
        grouping = GROUP_PHRASE.search(text) or WISE_PHRASE.search(text)
        before = text[:grouping.start()] if grouping else text

        for part in (before, text):
            entity = self._infer_count_entity(part)

            if entity is not None and entity != group_by:
                return entity

        measure = self._find_numeric_column_from_context(question)

        if measure is not None and measure != group_by:
            return measure

        return None

    # ---------------------------------------------------------
    # DATE FILTERS
    # ---------------------------------------------------------

    def _detect_date_filters(
        self,
        question: str,
    ) -> tuple[list[QueryFilter], list[str]]:
        """
        "in March", "March 2025", "Q1", "in 2024" -> filters on the
        datetime column the question refers to.
        """

        parts = detect_date_parts(question)

        if not parts:
            return [], []

        column = choose_date_column(question, self.df, self.schema)

        if column is None:
            raise ValueError(
                f"This sheet has no date column, so "
                f"'{parts[0].text}' can't be applied."
            )

        filters = [
            QueryFilter(column=column, operator=part.operator, value=part.value)
            for part in parts
        ]

        notes = []
        operators = {part.operator for part in parts}

        if "year" not in operators:

            dates = as_datetime(self.df[column])
            mask = dates.notna()

            for part in parts:
                mask &= date_part_mask(dates, part.operator, part.value)

            years = sorted(int(year) for year in dates[mask].dt.year.unique())

            if len(years) > 1:
                label = " ".join(
                    describe_date_part(part.operator, part.value)
                    for part in parts
                )
                listed = ", ".join(str(year) for year in years[:-1])
                notes.append(
                    f"{label} appears in {listed} and {years[-1]}, so all "
                    f"of them are included. Add a year (e.g. "
                    f"\"{label} {years[-1]}\") to narrow it down."
                )

        return filters, notes

    # ---------------------------------------------------------
    # ID-LIKE VALUES
    # ---------------------------------------------------------

    def _code_index(self) -> dict[tuple[str, int], list[tuple[str, Any]]]:
        """
        ID-like values keyed by (prefix, number), so "MR01" finds
        "MR001" and "cust-12" finds "CUST-0012".
        """

        if getattr(self, "_codes", None) is not None:
            return self._codes

        index: dict[tuple[str, int], list[tuple[str, Any]]] = {}

        for column in self.filterable_columns:

            values = self.df[column].dropna().unique()

            if len(values) == 0:
                continue

            keys = [(code_key(value), value) for value in values]

            if sum(key is not None for key, _ in keys) < 0.8 * len(values):
                continue

            for key, value in keys:
                if key is not None:
                    index.setdefault(key, []).append((column, value))

        self._codes = index

        return index

    def _detect_code_filters(
        self,
        question: str,
        existing: list[QueryFilter],
    ) -> tuple[list[QueryFilter], list[str]]:

        text = question.lower()

        already = {str(item.value).lower() for item in existing}

        filters = []
        unresolved = []

        for match in CODE_TOKEN.finditer(text):

            token = match.group(0)

            if token in already:
                continue

            if self._resolve_column_reference(token) is not None:
                continue

            prefix = match.group(1)
            hits = self._code_index().get((prefix, int(match.group(2))))

            if not hits:
                unresolved.append(question[match.start():match.end()])
                continue

            # Prefer the column named after the prefix (MR01 -> MRId).
            column, value = max(
                hits,
                key=lambda hit: prefix in name_tokens(hit[0]),
            )

            filters.append(QueryFilter(column=column, operator="=", value=value))

        return filters, unresolved

    # ---------------------------------------------------------
    # GLUED WORDS
    # ---------------------------------------------------------

    def _split_glued_words(self, question: str) -> str:
        """
        "doctorsId from Pune NorthTerritory" ->
        "doctors id from Pune north territory".

        Words that are themselves a column name or a value in the
        data ("InterestRate") are kept as typed.
        """

        def split(match: re.Match) -> str:
            word = match.group(0)
            if word.lower() in self._known_terms():
                return word
            return " ".join(name_tokens(word))

        return GLUED_WORD.sub(split, question)

    def _known_terms(self) -> set[str]:
        """Lowercase column names and categorical values."""

        if getattr(self, "_terms", None) is not None:
            return self._terms

        terms = {str(column).lower() for column in self.columns}

        for column in self.filterable_columns:
            terms.update(
                str(value).strip().lower()
                for value in self.df[column].dropna().unique()
            )

        self._terms = terms

        return terms
