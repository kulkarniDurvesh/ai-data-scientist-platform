"""
Example questions generated from the dataset schema.

They use the dataset's own column names, so they work for any dataset
and double as a guide to the phrasing the query parser understands.
"""

from __future__ import annotations

import pandas as pd

from core.schema_inference import (
    IDENTIFIER_NAME_TOKENS,
    DatasetSchema,
    name_tokens,
)


def suggest_questions(
    dataframe: pd.DataFrame,
    schema: DatasetSchema,
    limit: int = 6,
) -> list[str]:

    measures = _rank_measures(dataframe, schema)
    dimensions = [
        name
        for name in schema.dimensions
        if not schema.columns[name].is_numeric
    ]
    entity = _entity_column(schema)

    questions: list[str] = []

    if measures:
        questions.append(f"What is the average {measures[0]}?")

    if measures and dimensions:
        measure, dimension = measures[0], dimensions[0]
        questions.append(f"Show average {measure} by {dimension}.")
        questions.append(
            f"Which are the top 5 {_plural_phrase(dimension)} by {measure}?"
        )

    if entity:
        noun = _entity_noun(entity)
        questions.append(f"How many {noun} are there?")

        if dimensions:
            value = _frequent_value(dataframe, dimensions[0])
            if value is not None:
                questions.append(f"How many {noun} are there in {value}?")

    if len(measures) > 1 and dimensions:
        # Only additive quantities are summed; ratios are averaged.
        verb = "total" if schema.columns[measures[1]].kind == "count" else "average"
        questions.append(
            f"Show {verb} {measures[1]} by {dimensions[-1]}."
        )

    if len(measures) > 1:
        threshold = _round_threshold(dataframe[measures[1]])
        condition = f"where {measures[1]} > {threshold}"

        if threshold is not None and dimensions:
            questions.append(
                f"Show average {measures[0]} by {dimensions[0]} {condition}."
            )
        elif threshold is not None:
            questions.append(
                f"What is the average {measures[0]} {condition}?"
            )

    return list(dict.fromkeys(questions))[:limit]


def _rank_measures(dataframe: pd.DataFrame, schema: DatasetSchema) -> list[str]:
    """Measures with more distinct values first (more interesting to ask)."""

    return sorted(
        schema.measures,
        key=lambda name: -schema.columns[name].n_unique,
    )


def _entity_column(schema: DatasetSchema) -> str | None:
    for name in schema.identifiers:
        tokens = name_tokens(name)
        if len(tokens) > 1 and tokens[-1] in IDENTIFIER_NAME_TOKENS:
            return name
    return None


def _entity_noun(column: str) -> str:
    tokens = name_tokens(column)[:-1]
    return _pluralize(" ".join(tokens))


def _plural_phrase(column: str) -> str:
    return _pluralize(" ".join(name_tokens(column)))


def _pluralize(phrase: str) -> str:
    if phrase.endswith("y") and not phrase.endswith(("ay", "ey", "oy", "uy")):
        return phrase[:-1] + "ies"
    if phrase.endswith(("s", "x", "z", "ch", "sh")):
        return phrase + "es"
    return phrase + "s"


def _frequent_value(dataframe: pd.DataFrame, column: str):
    counts = dataframe[column].dropna().astype(str).value_counts()
    return counts.index[0] if len(counts) else None


def _round_threshold(series: pd.Series):
    values = series.dropna()

    if values.empty:
        return None

    median = float(values.median())

    if median == 0:
        return None

    if abs(median) >= 10:
        return int(round(median))

    return round(median, 2)
