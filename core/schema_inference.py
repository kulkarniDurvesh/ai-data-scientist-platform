"""
Single source of truth for column semantics.

Every other module (EDA pipeline, chart recommender, manual builder,
NL query layer, dashboard) asks this module what a column *is* instead
of re-detecting it. Nothing here depends on a specific dataset: the
rules use dtypes, cardinality, value shape and generic English naming
conventions only.

Roles
-----
identifier : keys / labels of an entity (CustomerId, OrderCode, Phone)
dimension  : categorical column usable for grouping
measure    : numeric column usable for aggregation
binary     : exactly two values (flags, 0/1 targets, Yes/No)
time       : datetime column
text       : free text
constant   : a single distinct value
empty      : no values at all
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


ROLES = (
    "identifier",
    "dimension",
    "measure",
    "binary",
    "time",
    "text",
    "constant",
    "empty",
)

# Generic naming conventions for key-like columns.
IDENTIFIER_NAME_TOKENS = {
    "id", "uuid", "guid", "key", "code", "sku",
}

# Trailing words that name an entity on a column already known to be a
# key ("ShipmentNo" -> shipment). Too common to mark keys on their own.
ENTITY_KEY_TOKENS = IDENTIFIER_NAME_TOKENS | {"no", "num", "number"}

# Generic naming conventions for contact / reference values that look
# numeric but must never be aggregated.
REFERENCE_NAME_TOKENS = {
    "phone", "mobile", "email", "zip", "zipcode", "pincode",
    "postcode", "postal", "fax", "url", "ip",
}


@dataclass
class SchemaConfig:
    max_dimension_cardinality: int = 50
    identifier_unique_ratio: float = 0.95
    text_min_avg_length: float = 40.0
    min_time_points: int = 3


@dataclass
class ColumnSchema:
    name: str
    dtype: str
    role: str
    n_rows: int
    n_unique: int
    unique_ratio: float
    missing_ratio: float
    is_numeric: bool
    is_datetime: bool
    # measure: count | continuous | ratio | binary
    # dimension: low | high (cardinality)
    kind: str | None = None
    high_cardinality: bool = False
    sample_values: list[Any] = field(default_factory=list)
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "column": self.name,
            "dtype": self.dtype,
            "role": self.role,
            "kind": self.kind,
            "unique": self.n_unique,
            "unique_pct": round(self.unique_ratio * 100, 2),
            "missing_pct": round(self.missing_ratio * 100, 2),
            "high_cardinality": self.high_cardinality,
            "sample_values": self.sample_values,
            "reason": self.reason,
        }


@dataclass
class DatasetSchema:
    columns: dict[str, ColumnSchema]
    n_rows: int
    primary_time: str | None = None
    config: SchemaConfig = field(default_factory=SchemaConfig)

    # ------------------------------------------------------------------
    # Role accessors
    # ------------------------------------------------------------------

    def by_role(self, *roles: str) -> list[str]:
        return [
            name
            for name, column in self.columns.items()
            if column.role in roles
        ]

    @property
    def measures(self) -> list[str]:
        return self.by_role("measure")

    @property
    def dimensions(self) -> list[str]:
        """Low-cardinality dimensions (safe for automatic grouping)."""
        return [
            name
            for name in self.by_role("dimension")
            if not self.columns[name].high_cardinality
        ]

    @property
    def all_dimensions(self) -> list[str]:
        return self.by_role("dimension")

    @property
    def binaries(self) -> list[str]:
        return self.by_role("binary")

    @property
    def numeric_binaries(self) -> list[str]:
        """Binary 0/1 columns: their mean is a rate."""
        return [
            name
            for name in self.binaries
            if self.columns[name].is_numeric
        ]

    @property
    def times(self) -> list[str]:
        return self.by_role("time")

    @property
    def identifiers(self) -> list[str]:
        return self.by_role("identifier")

    @property
    def groupable(self) -> list[str]:
        """Columns that make sense on a categorical axis."""
        return self.dimensions + self.binaries

    @property
    def aggregatable(self) -> list[str]:
        """Numeric columns that can be summed / averaged."""
        return self.measures + self.numeric_binaries

    def role_of(self, column: str | None) -> str | None:
        if column is None or column not in self.columns:
            return None
        return self.columns[column].role

    def is_numeric(self, column: str | None) -> bool:
        return (
            column in self.columns
            and self.columns[column].is_numeric
        )

    def to_records(self) -> list[dict[str, Any]]:
        return [
            column.to_dict()
            for column in self.columns.values()
        ]


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------

def infer_schema(
    dataframe: pd.DataFrame,
    config: SchemaConfig | None = None,
) -> DatasetSchema:

    if dataframe is None:
        raise ValueError("DataFrame cannot be None.")

    config = config or SchemaConfig()

    columns = {
        str(name): _infer_column(dataframe[name], config)
        for name in dataframe.columns
    }

    schema = DatasetSchema(
        columns=columns,
        n_rows=len(dataframe),
        config=config,
    )

    schema.primary_time = _choose_primary_time(
        dataframe,
        schema,
    )

    return schema


def name_tokens(column_name: Any) -> list[str]:
    """
    Split a column name into lowercase word tokens.

        "CustomerID"       -> ["customer", "id"]
        "days_since_order" -> ["days", "since", "order"]
        "TotalSales"      -> ["total", "sales"]
    """

    text = str(column_name)
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    text = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", text)

    return [
        token
        for token in re.split(r"[^A-Za-z0-9]+", text.lower())
        if token
    ]


def same_word(first: str, second: str) -> bool:
    """
    Loose word match for names: "visited" ~ "Visits",
    "doctors" ~ "Doctor", "territories" ~ "territory".
    """

    a, b = _stem(first.lower()), _stem(second.lower())

    if a == b:
        return True

    shorter, longer = sorted((a, b), key=len)
    return len(shorter) >= 4 and longer.startswith(shorter)


def _stem(word: str) -> str:
    for suffix in ("ing", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def key_entity_name(column_name: Any) -> str | None:
    """
    The entity a key column names, from its trailing key word.

        "DoctorId"       -> "doctor"
        "sales_rep_code" -> "sales rep"
        "ShipmentNo"     -> "shipment"
        "AmountPaid"     -> None   (no key word, despite ending in "id")
    """

    tokens = name_tokens(column_name)

    if len(tokens) > 1 and tokens[-1] in ENTITY_KEY_TOKENS:
        return " ".join(tokens[:-1])

    return None


def is_identifier(column: Any, schema: "DatasetSchema | None" = None) -> bool:
    """
    True for key / reference columns. Uses the inferred role when a
    schema is available, otherwise the column's naming convention.
    """

    if schema is not None and str(column) in schema.columns:
        return schema.role_of(str(column)) == "identifier"

    tokens = name_tokens(column)
    return len(tokens) > 1 and tokens[-1] in IDENTIFIER_NAME_TOKENS


def humanize(column_name: Any) -> str:
    """Readable label for a column name: 'OrderRate' -> 'Order Rate'."""

    if column_name is None:
        return ""

    text = str(column_name)
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    text = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", text)
    text = re.sub(r"[_\-]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    if text.islower():
        text = text.title()

    return text


def format_number(value: Any) -> str:
    """Compact human-readable number: 1,234,567 / 12.35 / 0.0421."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)

    if not np.isfinite(number):
        return str(value)

    if number == int(number) and abs(number) < 1e15:
        return f"{int(number):,}"

    if abs(number) >= 1000:
        return f"{number:,.0f}"

    if abs(number) >= 1:
        return f"{number:,.2f}"

    return f"{number:.3g}"


def suggest_time_grain(series: pd.Series) -> str | None:
    """
    Pick a time bucket that yields a readable number of points.
    Returns None when the values are already at a coarse grain.
    """

    values = pd.to_datetime(series, errors="coerce").dropna()

    if values.empty:
        return None

    if values.nunique() <= 60:
        return None

    span_days = (values.max() - values.min()).days

    if span_days <= 90:
        return "D"
    if span_days <= 730:
        return "W"
    if span_days <= 365 * 6:
        return "M"
    if span_days <= 365 * 20:
        return "Q"
    return "Y"


# ----------------------------------------------------------------------
# Column inference
# ----------------------------------------------------------------------

def _infer_column(
    series: pd.Series,
    config: SchemaConfig,
) -> ColumnSchema:

    n_rows = len(series)
    non_null = series.dropna()

    n_unique = int(non_null.nunique())
    unique_ratio = n_unique / n_rows if n_rows else 0.0
    missing_ratio = 1 - (len(non_null) / n_rows) if n_rows else 0.0

    is_bool = pd.api.types.is_bool_dtype(series)
    is_numeric = (
        pd.api.types.is_numeric_dtype(series)
        and not is_bool
    )
    is_datetime = pd.api.types.is_datetime64_any_dtype(series)

    column = ColumnSchema(
        name=str(series.name),
        dtype=str(series.dtype),
        role="dimension",
        n_rows=n_rows,
        n_unique=n_unique,
        unique_ratio=unique_ratio,
        missing_ratio=missing_ratio,
        is_numeric=is_numeric,
        is_datetime=is_datetime,
        sample_values=_sample_values(non_null),
    )

    tokens = set(name_tokens(series.name))

    # -- empty / constant ------------------------------------------------

    if len(non_null) == 0:
        return _set(column, "empty", reason="No non-null values.")

    if n_unique == 1:
        return _set(column, "constant", reason="Only one distinct value.")

    # -- time ------------------------------------------------------------

    if is_datetime:
        return _set(column, "time", reason="Datetime values.")

    # -- binary ----------------------------------------------------------

    if is_bool or n_unique == 2:
        return _set(
            column,
            "binary",
            kind="binary",
            reason="Exactly two distinct values.",
        )

    # -- identifiers by naming convention ---------------------------------

    looks_like_key = bool(tokens & IDENTIFIER_NAME_TOKENS) and (
        _ends_with_key_token(series.name) or not is_numeric
    )
    looks_like_reference = bool(tokens & REFERENCE_NAME_TOKENS)

    if looks_like_reference:
        return _set(
            column,
            "identifier",
            reason="Name indicates a contact/reference value.",
        )

    if looks_like_key and not _is_fractional(non_null, is_numeric):
        return _set(
            column,
            "identifier",
            reason="Name indicates a key or code.",
        )

    # -- numeric -----------------------------------------------------------

    if is_numeric:

        if _is_row_index(non_null, n_rows):
            return _set(
                column,
                "identifier",
                reason="Sequential unique integers (row index).",
            )

        return _set(
            column,
            "measure",
            kind=_measure_kind(non_null),
            reason="Numeric values.",
        )

    # -- strings -----------------------------------------------------------

    as_text = non_null.astype(str)
    avg_length = float(as_text.str.len().mean())

    if avg_length >= config.text_min_avg_length:
        return _set(column, "text", reason="Long free-text values.")

    if n_unique <= config.max_dimension_cardinality:
        return _set(
            column,
            "dimension",
            kind="low",
            reason=f"{n_unique} categories.",
        )

    if unique_ratio >= config.identifier_unique_ratio:
        return _set(
            column,
            "identifier",
            reason="Almost every value is unique.",
        )

    has_spaces = float(as_text.str.contains(r"\s").mean())

    if has_spaces > 0.8 and avg_length >= 25:
        return _set(column, "text", reason="Sentence-like values.")

    return _set(
        column,
        "dimension",
        kind="high",
        high_cardinality=True,
        reason=f"{n_unique} categories (high cardinality).",
    )


def _set(
    column: ColumnSchema,
    role: str,
    kind: str | None = None,
    high_cardinality: bool = False,
    reason: str = "",
) -> ColumnSchema:
    column.role = role
    column.kind = kind
    column.high_cardinality = high_cardinality
    column.reason = reason
    return column


def _ends_with_key_token(column_name: Any) -> bool:
    tokens = name_tokens(column_name)
    return bool(tokens) and (
        tokens[-1] in IDENTIFIER_NAME_TOKENS
        or tokens[0] in IDENTIFIER_NAME_TOKENS
    )


def _is_fractional(non_null: pd.Series, is_numeric: bool) -> bool:
    if not is_numeric:
        return False
    values = non_null.astype(float)
    return bool((values % 1 != 0).any())


def _is_row_index(non_null: pd.Series, n_rows: int) -> bool:
    if len(non_null) != n_rows or n_rows < 20:
        return False

    if not pd.api.types.is_integer_dtype(non_null):
        return False

    diffs = np.diff(non_null.to_numpy())

    return bool(len(diffs) and np.all(diffs == 1))


def _measure_kind(non_null: pd.Series) -> str:
    values = non_null.astype(float)

    if values.min() >= 0 and values.max() <= 1:
        return "ratio"

    # Counts are small non-negative whole numbers (sales, items, units).
    # Large integers such as scores, prices or amounts are treated as
    # continuous so they default to averages rather than sums.
    if (
        bool((values % 1 == 0).all())
        and values.min() >= 0
        and values.quantile(0.95) <= 1000
    ):
        return "count"

    return "continuous"


def _sample_values(non_null: pd.Series, limit: int = 3) -> list[Any]:
    values = []

    for value in non_null.unique()[:limit]:
        if isinstance(value, (pd.Timestamp, np.datetime64)):
            values.append(str(pd.Timestamp(value).date()))
        elif isinstance(value, (np.integer, np.floating)):
            values.append(value.item())
        else:
            text = str(value)
            values.append(text if len(text) <= 40 else text[:37] + "...")

    return values


def _choose_primary_time(
    dataframe: pd.DataFrame,
    schema: DatasetSchema,
) -> str | None:
    """
    The primary time axis is the datetime column with good coverage and
    the coarsest regular grain (e.g. a 'month' column beats a raw
    'last activity timestamp').
    """

    candidates = []

    for name in schema.times:
        column = schema.columns[name]

        if column.missing_ratio > 0.5:
            continue

        if column.n_unique < schema.config.min_time_points:
            continue

        candidates.append(
            (column.n_unique, column.missing_ratio, name)
        )

    if not candidates:
        return None

    candidates.sort()

    return candidates[0][2]
