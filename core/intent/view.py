"""
What the goal box can see of a dataset: its tables and their columns.

A DataView wraps a callback that returns (frame, schema, sources, links)
for a sheet, as the session layer provides, so this package needs no
knowledge of the dashboard or the API. Which tables suit which task is
computed here with the same option functions the tabs use.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd

from core.forecast import date_options
from core.recommend import default_roles
from core.schema_inference import DatasetSchema
from core.segment import feature_options

MIN_ROWS = 20
MAX_VALUES_PER_COLUMN = 1000

Context = tuple[pd.DataFrame, DatasetSchema, dict[str, str], list]


@dataclass
class DataView:
    name: str
    sheets: list[str | None]
    model_sheet: str | None
    context_of: Callable[[str | None], Context]
    _contexts: dict[Any, Context] = field(default_factory=dict, repr=False)
    _suits: dict[tuple[str, Any], bool] = field(default_factory=dict, repr=False)
    _values: dict[Any, set[str]] = field(default_factory=dict, repr=False)

    def context(self, sheet: str | None) -> Context:
        if sheet not in self._contexts:
            self._contexts[sheet] = self.context_of(sheet)
        return self._contexts[sheet]

    def label(self, sheet: str | None) -> str:
        return sheet if sheet else self.name

    def native_columns(self, sheet: str | None) -> list[str]:
        """The sheet's own columns (not those added from linked sheets)."""

        frame, _, sources, _ = self.context(sheet)
        return [c for c in frame.columns if sources.get(c) in (None, sheet)]

    def sheets_for(self, task: str) -> list[str | None]:
        """Tables that can serve a task, the best first."""

        if task in ("rank", "classify", "regress", "ask"):
            return [self.model_sheet]
        found = [sheet for sheet in self.sheets if self._suits_task(task, sheet)]
        if task == "recommend":
            found.sort(key=lambda sheet: -len(self.context(sheet)[0]))
        else:
            found.sort(key=lambda sheet: sheet != self.model_sheet)
        return found

    def _suits_task(self, task: str, sheet: str | None) -> bool:
        key = (task, sheet)
        if key not in self._suits:
            self._suits[key] = self._check(task, sheet)
        return self._suits[key]

    def _check(self, task: str, sheet: str | None) -> bool:
        try:
            frame, schema, sources, links = self.context(sheet)
        except (ValueError, KeyError, TypeError):
            return False
        if len(frame) < MIN_ROWS:
            return False
        try:
            if task == "recommend":
                default_roles(frame, schema, sources, links)
                return True
            if task in ("forecast", "why"):
                return bool(date_options(frame))
            if task == "segment":
                return bool(feature_options(frame, schema, None)[0])
        except (ValueError, KeyError, TypeError):
            return False
        return False

    def value_words(self, sheet: str | None) -> set[str]:
        """Lowercase words that appear in the sheet's category values."""

        if sheet not in self._values:
            frame, schema, _, _ = self.context(sheet)
            words: set[str] = set()
            for column in frame.columns:
                if schema.role_of(column) in ("measure", "time"):
                    continue
                values = frame[column].dropna()
                if values.nunique() > MAX_VALUES_PER_COLUMN:
                    continue
                for value in values.astype(str).unique():
                    words.update(w for w in _words(value) if len(w) >= 3)
            self._values[sheet] = words
        return self._values[sheet]


def _words(text: str) -> list[str]:
    import re

    return re.findall(r"[a-z]+", str(text).lower())
