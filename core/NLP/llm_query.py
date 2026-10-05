"""
Language-model fallback for the Ask box (Phase 6b).

When the rule parser can't read a question ("how much did we make in the
south?"), a language model proposes the same structure the parser makes
(intent, column, filters, grouping ...). The proposal is validated here
against the real columns and values, then planned and executed by the
existing planner and engine, so the model never produces a number and
never filters on a value that isn't in the data.
"""

from __future__ import annotations

from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, Field

from core.llm import Provider, structured
from core.schema_inference import DatasetSchema

from .query_schema import ParsedQuery, QueryFilter

MAX_COLUMNS = 60
SAMPLES = 4
MAX_CATEGORY_VALUES = 5000

Intent = Literal["COUNT", "SUM", "AVERAGE", "MIN", "MAX", "FILTER", "TOP_N", "BOTTOM_N", "CHART"]


class Unanswerable(ValueError):
    """The model judged that the table can't answer the question."""


class ProposedFilter(BaseModel):
    column: str
    operator: Literal["=", "!=", ">", "<", ">=", "<="] = "="
    value: str | float | int


class QueryProposal(BaseModel):
    answerable: bool = Field(True, description="False if the columns can't answer the question")
    intent: Intent | None = None
    entity: str | None = Field(None, description="Column the operation applies to (counted, summed, listed ...)")
    filters: list[ProposedFilter] = Field(default_factory=list)
    group_by: str | None = Field(None, description="Column to group by, if the question asks 'by' or 'per'")
    subject: str | None = Field(None, description="For 'which X has the most Y': the X column")
    aggregation: Literal["SUM", "AVERAGE", "COUNT", "MIN", "MAX"] | None = None
    limit: int | None = None
    chart_type: Literal["BAR", "LINE", "PIE", "SCATTER", "HISTOGRAM"] | None = None
    reason: str | None = Field(None, description="Why it can't be answered, when answerable is false")


SYSTEM = """You translate a question about one table into a structured query. You never answer it yourself.
Intents: COUNT (how many distinct values of entity), SUM / AVERAGE / MIN / MAX (of a numeric entity;
with group_by for one value per group), FILTER (list the entity values, or rows, that match the filters),
TOP_N / BOTTOM_N (the `limit` values of subject with the highest / lowest numeric entity),
CHART (entity by group_by).
For "which X has the highest total Y" use SUM with entity=Y and group_by=X.
Rules:
- Use column names exactly as listed; map synonyms to them. Never invent columns.
- Filter values must be values of that column, as listed (keep their spelling); numbers for numeric columns.
- If the table can't answer the question, set answerable to false and give a short reason.
Reply with JSON only."""


def describe_table(frame: pd.DataFrame, schema: DatasetSchema) -> str:
    """Column names, roles and a few values: all the model sees of the data."""

    parts = []
    for column in list(frame.columns)[:MAX_COLUMNS]:
        role = schema.role_of(column)
        if role in ("constant", "empty"):
            continue
        entry = f"{column}:{role}"
        if role in ("dimension", "binary", "identifier"):
            values = frame[column].dropna().astype(str).unique()[:SAMPLES]
            entry += " (" + "/".join(value[:18] for value in values) + ")"
        parts.append(entry)
    return "Columns: " + ", ".join(parts)


def propose_query(question: str, frame: pd.DataFrame, schema: DatasetSchema, provider: Provider) -> QueryProposal:
    user = f"{describe_table(frame, schema)}\n\nQuestion: {question.strip()}"
    return structured(provider, SYSTEM, user, QueryProposal)


def to_parsed_query(proposal: QueryProposal, frame: pd.DataFrame, schema: DatasetSchema) -> ParsedQuery:
    """A validated ParsedQuery, or ValueError naming what doesn't exist."""

    from core.intent import match_column

    if not proposal.answerable or proposal.intent is None:
        raise Unanswerable(proposal.reason or "The data has no columns that answer this question.")

    columns = list(frame.columns)

    def column(name: str | None, label: str) -> str | None:
        if name is None:
            return None
        found = match_column(name, columns)
        if found is None:
            raise ValueError(f"The {label} '{name}' is not a column of this table.")
        return found

    filters = []
    for item in proposal.filters:
        name = column(item.column, "filter column")
        filters.append(QueryFilter(column=name, operator=item.operator, value=_value(frame, schema, name, item)))

    return ParsedQuery(
        intent=proposal.intent,
        entity=column(proposal.entity, "column"),
        filters=filters,
        subject=column(proposal.subject, "column"),
        limit=proposal.limit,
        group_by=column(proposal.group_by, "grouping column"),
        aggregation=proposal.aggregation,
        chart_type=proposal.chart_type,
        notes=[],
    )


def _value(frame: pd.DataFrame, schema: DatasetSchema, column: str, item: ProposedFilter) -> Any:
    series = frame[column]
    if schema.is_numeric(column) and schema.role_of(column) != "binary":
        try:
            return float(item.value)
        except (TypeError, ValueError):
            raise ValueError(f"'{item.value}' is not a number, but {column} is numeric.") from None
    if pd.api.types.is_datetime64_any_dtype(series):
        raise ValueError("Date filters are answered by the rules; name the month or year in the question.")

    values = series.dropna()
    if values.nunique() > MAX_CATEGORY_VALUES:
        raise ValueError(f"{column} has too many values to check '{item.value}'.")
    lookup = {str(value).strip().lower(): value for value in values.unique()}
    wanted = str(item.value).strip().lower()
    if wanted.endswith(".0") and wanted[:-2] in lookup:
        wanted = wanted[:-2]
    if wanted not in lookup:
        raise ValueError(f"'{item.value}' is not a value of {column}.")
    return lookup[wanted]


def describe_query(parsed: ParsedQuery) -> str:
    """How the model read the question, in the data's own words."""

    parts = [parsed.intent.replace("_", " ").lower()]
    if parsed.entity:
        parts.append(f"of '{parsed.entity}'")
    if parsed.subject:
        parts.append(f"per '{parsed.subject}'")
    if parsed.group_by:
        parts.append(f"by '{parsed.group_by}'")
    if parsed.filters:
        parts.append("where " + " and ".join(f"{f.column} {f.operator} {f.value}" for f in parsed.filters))
    return " ".join(parts)
