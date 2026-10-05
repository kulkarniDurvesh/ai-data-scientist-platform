"""
Calendar expressions in questions ("in March", "March 2025", "Q1",
"in 2024") and the matching filters on any datetime column.

Nothing here knows a dataset: the date column is picked by data type,
then by how well its name matches the words of the question.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import pandas as pd

from core.schema_inference import DatasetSchema, name_tokens, same_word


# Filter operators that compare one calendar part of a date column.
DATE_PART_OPERATORS = {"month", "quarter", "year"}

MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5,
    "june": 6, "july": 7, "august": 8, "september": 9, "october": 10,
    "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7,
    "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}

MONTH_NAMES = [
    "January", "February", "March", "April", "May", "June", "July",
    "August", "September", "October", "November", "December",
]

# Words that name a calendar unit, not a column ("in the month of March").
DATE_UNIT_WORDS = {
    "day", "days", "week", "weeks", "month", "months", "quarter",
    "quarters", "year", "years", "date", "dates", "time", "period",
}

_MONTH = "|".join(sorted(MONTHS, key=len, reverse=True))
_YEAR = r"(?:19|20)\d{2}"

_MONTH_YEAR = re.compile(rf"\b({_MONTH})\.?\s*,?\s*({_YEAR})\b")
_QUARTER = re.compile(rf"\bq([1-4])\b(?:\s*,?\s*({_YEAR})\b)?")
_MONTH_ONLY = re.compile(rf"\b({_MONTH})\b")
# A bare year only counts after a word that introduces a period, so
# "Sales more than 2000" stays a numeric filter.
_YEAR_ONLY = re.compile(rf"\b(?:in|of|during|for|year|since)\s+({_YEAR})\b")

# "may" is also a verb; accept it only in a clear date context.
_AMBIGUOUS_MONTHS = {"may"}
_DATE_CONTEXT = re.compile(r"\b(?:in|of|during|for|since|month)\s+$")


@dataclass
class DatePart:
    operator: str  # month | quarter | year
    value: int
    text: str


def detect_date_parts(question: str) -> list[DatePart]:
    """Calendar expressions in the question, without overlaps."""

    text = question.lower()
    taken: list[tuple[int, int]] = []
    parts: list[DatePart] = []

    def free(start: int, end: int) -> bool:
        return all(end <= s or start >= e for s, e in taken)

    for match in _MONTH_YEAR.finditer(text):
        taken.append(match.span())
        parts.append(DatePart("month", MONTHS[match.group(1)], match.group(0)))
        parts.append(DatePart("year", int(match.group(2)), match.group(2)))

    for match in _QUARTER.finditer(text):
        if not free(*match.span()):
            continue
        taken.append(match.span())
        parts.append(DatePart("quarter", int(match.group(1)), match.group(0)))
        if match.group(2):
            parts.append(DatePart("year", int(match.group(2)), match.group(2)))

    for match in _MONTH_ONLY.finditer(text):
        if not free(*match.span()):
            continue
        word = match.group(1)
        if word in _AMBIGUOUS_MONTHS and not _DATE_CONTEXT.search(text[: match.start()]):
            continue
        taken.append(match.span())
        parts.append(DatePart("month", MONTHS[word], word))

    for match in _YEAR_ONLY.finditer(text):
        span = match.span(1)
        if not free(*span):
            continue
        taken.append(span)
        parts.append(DatePart("year", int(match.group(1)), match.group(1)))

    return parts


def datetime_columns(dataframe: pd.DataFrame) -> list[str]:
    return [
        column
        for column in dataframe.columns
        if pd.api.types.is_datetime64_any_dtype(dataframe[column])
    ]


def choose_date_column(
    question: str,
    dataframe: pd.DataFrame,
    schema: DatasetSchema | None = None,
) -> str | None:
    """
    The datetime column a date expression refers to.

    Columns whose name shares a word with the question win
    ("doctors visited in March" -> LastVisitDate); otherwise the
    dataset's primary time column, then the first datetime column.
    """

    candidates = datetime_columns(dataframe)

    if not candidates:
        return None

    words = [
        word
        for word in re.findall(r"[a-z]+", question.lower())
        if len(word) >= 3 and word not in DATE_UNIT_WORDS
    ]

    primary = schema.primary_time if schema is not None else None

    def score(column: str) -> tuple[int, bool, float]:
        hits = 0
        for token in name_tokens(column):
            if token in DATE_UNIT_WORDS or len(token) < 3:
                continue
            if any(same_word(word, token) for word in words):
                hits += 1
        filled = float(dataframe[column].notna().mean())
        return hits, column == primary, filled

    return max(candidates, key=score)


def as_datetime(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    return pd.to_datetime(series, errors="coerce")


def date_part_mask(series: pd.Series, operator: str, value: Any) -> pd.Series:
    dates = as_datetime(series)
    value = int(value)

    if operator == "month":
        return dates.dt.month == value
    if operator == "quarter":
        return dates.dt.quarter == value
    if operator == "year":
        return dates.dt.year == value

    raise ValueError(f"Unsupported date filter: '{operator}'")


def describe_date_part(operator: str, value: Any) -> str:
    value = int(value)
    if operator == "month":
        return MONTH_NAMES[value - 1]
    if operator == "quarter":
        return f"Q{value}"
    return str(value)
