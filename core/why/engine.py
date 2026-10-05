"""
Investigation: compare, explain, find unusual groups, rank attention.

    table + ChangeSpec (+ an entity dimension for attention)
        -> comparison of the latest complete period with the chosen one
        -> drill-down explanation chain
        -> contribution tables for every dimension
        -> total over time
        -> unusual groups and attention ranking for the entity dimension
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

from core.forecast.series import ROW_COUNT
from core.schema_inference import DatasetSchema

from .attention import attention_ranking, unusual_groups
from .change import (
    Breakdown,
    ChangeSpec,
    Step,
    all_breakdowns,
    complete_periods,
    explain_change,
    measure_label,
    periods,
    value,
)

Progress = Callable[[str], None]

TIME_OF_DAY = r"^\s*\d{1,2}:\d{2}(:\d{2})?\s*$"

# Dimensions with more levels than this are offered for attention only.
MAX_DRILL_LEVELS = 100
MAX_ATTENTION_LEVELS = 2000


@dataclass
class Investigation:
    spec: ChangeSpec
    steps: list[Step]
    breakdowns: list[Breakdown]
    trend: pd.Series
    attention_dimension: str | None
    attention: pd.DataFrame
    unusual: pd.DataFrame
    higher_is_better: bool = True
    notes: list[str] = field(default_factory=list)
    seconds: float = 0.0

    @property
    def comparison(self):
        return self.steps[0].comparison

    @property
    def summary(self) -> str:
        return " ".join(step.text for step in self.steps)

    def label(self) -> str:
        return measure_label(self.spec)


def drill_options(dataframe: pd.DataFrame, schema: DatasetSchema) -> list[str]:
    """
    Columns that can split a total: categories and keys with few levels.
    Times of day are skipped, and of columns that map one-to-one onto each
    other (a key, its name, its e-mail) only the first is kept.
    """

    options = [
        column for column in dataframe.columns
        if schema.role_of(column) in {"dimension", "binary", "identifier"}
        and 2 <= dataframe[column].nunique() <= MAX_DRILL_LEVELS
        and not _times_of_day(dataframe[column])
    ]
    return sorted(_drop_equivalent(dataframe, options), key=lambda column: dataframe[column].nunique())


def _times_of_day(series: pd.Series) -> bool:
    text = series.dropna().astype(str).head(200)
    return bool(len(text)) and text.str.match(TIME_OF_DAY).mean() > 0.8


def _drop_equivalent(dataframe: pd.DataFrame, columns: list[str]) -> list[str]:
    kept: list[str] = []
    for column in columns:
        duplicate = False
        for other in kept:
            both = dataframe[[column, other]].dropna()
            if len(both) and both.groupby(column)[other].nunique().max() <= 1 and both.groupby(other)[column].nunique().max() <= 1:
                duplicate = True
                break
        if not duplicate:
            kept.append(column)
    return kept


def attention_options(dataframe: pd.DataFrame, schema: DatasetSchema) -> list[str]:
    """Entities to rank: keys first, then categories."""

    # Keys unique to almost every row (one row per visit) have no history to rank.
    options = [
        column for column in dataframe.columns
        if schema.role_of(column) in {"identifier", "dimension"}
        and 2 <= dataframe[column].nunique() <= min(MAX_ATTENTION_LEVELS, len(dataframe) * 0.5)
    ]
    return sorted(options, key=lambda column: (schema.role_of(column) != "identifier", -dataframe[column].nunique()))


def investigate(
    dataframe: pd.DataFrame,
    spec: ChangeSpec,
    attention_dimension: str | None = None,
    higher_is_better: bool = True,
    progress: Progress | None = None,
) -> Investigation:

    started = time.time()
    say = progress or (lambda message: None)
    frame = dataframe.dropna(subset=[spec.time])

    if spec.measure != ROW_COUNT and spec.measure not in frame.columns:
        raise ValueError(f"Column '{spec.measure}' does not exist.")

    say("Comparing periods")
    steps = explain_change(frame, spec)

    say("Breaking the change down by every dimension")
    breakdowns = all_breakdowns(frame, spec)

    stamps = complete_periods(frame, spec)
    stamp = periods(frame, spec)
    trend = pd.Series({p: value(frame[stamp == p], spec) for p in stamps})

    attention = pd.DataFrame()
    unusual = pd.DataFrame()
    if attention_dimension:
        say(f"Ranking {attention_dimension} by attention")
        attention = attention_ranking(frame, spec, attention_dimension, higher_is_better)
        unusual = unusual_groups(frame, spec, attention_dimension)

    notes = []
    if spec.dimensions and not any(step.breakdown for step in steps):
        notes.append("No dimension splits this change into groups; the total moved evenly.")

    say("Done")
    return Investigation(
        spec=spec, steps=steps, breakdowns=breakdowns, trend=trend,
        attention_dimension=attention_dimension, attention=attention, unusual=unusual,
        higher_is_better=higher_is_better, notes=notes,
        seconds=round(time.time() - started, 1),
    )
