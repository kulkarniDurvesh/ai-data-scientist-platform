"""
Several sheets of one file, linked through shared ID columns.

    Visits.DoctorId  ->  Doctors.DoctorId   (Doctors has one row per id)
    Visits.MRId      ->  MRs.MRId           -> MRs.UserId -> Users.UserId

A link is found from the data alone: a key column (identifier role) in
one sheet whose name matches a column in another sheet where every
value is unique and most of the key's values appear. No sheet or
column names are configured anywhere.

`enriched(sheet)` returns the sheet with the columns of the sheets it
links to added on, so a question asked on Visits can filter by a
doctor's Specialty or list doctor names.
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field

import pandas as pd

from core.loader import load_all_sheets
from core.schema_inference import DatasetSchema, infer_schema, is_identifier


# Share of a key's distinct values that must exist in the lookup sheet.
MIN_LINK_OVERLAP = 0.5
# How many links deep enrichment follows (Visits -> MRs -> Users).
MAX_LINK_DEPTH = 2


@dataclass(frozen=True)
class Link:
    sheet: str
    column: str
    lookup_sheet: str
    lookup_column: str


@dataclass
class EnrichedSheet:
    sheet: str
    df: pd.DataFrame
    # Added column -> the sheet it was looked up from.
    sources: dict[str, str] = field(default_factory=dict)

    def sheets_used(self, columns) -> list[str]:
        """Lookup sheets that the given columns came from, in order."""

        used = []
        for column in columns:
            sheet = self.sources.get(column)
            if sheet and sheet not in used:
                used.append(sheet)
        return used


class Workbook:

    def __init__(self, content: bytes, filename: str, sheets: list[str]):
        self.content = content
        self.filename = filename
        self.sheets = list(sheets)

        self._loaded: dict[str, pd.DataFrame] | None = None
        self._schemas: dict[str, DatasetSchema] = {}
        self._links: list[Link] | None = None
        self._enriched: dict[str, EnrichedSheet] = {}
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Sheets
    # ------------------------------------------------------------------

    def frame(self, sheet: str) -> pd.DataFrame | None:
        with self._lock:
            if self._loaded is None:
                try:
                    self._loaded = load_all_sheets(self.content, self.filename)
                except Exception:  # noqa: BLE001 - an unreadable file has no sheets
                    self._loaded = {}
            return self._loaded.get(sheet)

    def schema(self, sheet: str) -> DatasetSchema | None:
        with self._lock:
            if sheet not in self._schemas:
                frame = self.frame(sheet)
                if frame is None or frame.empty:
                    return None
                self._schemas[sheet] = infer_schema(frame)
            return self._schemas[sheet]

    def readable_sheets(self) -> list[str]:
        return [
            sheet
            for sheet in self.sheets
            if self.frame(sheet) is not None and not self.frame(sheet).empty
        ]

    # ------------------------------------------------------------------
    # Links
    # ------------------------------------------------------------------

    def links(self) -> list[Link]:
        with self._lock:
            if self._links is None:
                self._links = self._detect_links()
            return self._links

    def _detect_links(self) -> list[Link]:
        sheets = self.readable_sheets()

        # Columns whose values are unique within their sheet, by name.
        lookups: dict[str, list[tuple[str, str]]] = {}

        for sheet in sheets:
            frame = self.frame(sheet)
            for column in frame.columns:
                values = frame[column].dropna()
                if len(values) and values.is_unique:
                    lookups.setdefault(_normalize(column), []).append((sheet, column))

        links = []

        for sheet in sheets:
            frame = self.frame(sheet)
            schema = self.schema(sheet)

            for column in frame.columns:
                if not is_identifier(column, schema):
                    continue

                keys = set(_as_keys(frame[column]).dropna().unique())

                if not keys:
                    continue

                for lookup_sheet, lookup_column in lookups.get(_normalize(column), []):
                    if lookup_sheet == sheet:
                        continue

                    lookup_keys = set(_as_keys(self.frame(lookup_sheet)[lookup_column]))
                    overlap = len(keys & lookup_keys) / len(keys)

                    if overlap >= MIN_LINK_OVERLAP:
                        links.append(Link(sheet, column, lookup_sheet, lookup_column))

        return links

    # ------------------------------------------------------------------
    # Enrichment
    # ------------------------------------------------------------------

    def enriched(self, sheet: str) -> EnrichedSheet | None:
        with self._lock:
            if sheet not in self._enriched:
                base = self.frame(sheet)
                if base is None:
                    return None
                self._enriched[sheet] = self._enrich(sheet, base)
            return self._enriched[sheet]

    def _enrich(self, sheet: str, base: pd.DataFrame) -> EnrichedSheet:
        """
        Add the columns of linked lookup sheets, following links up to
        MAX_LINK_DEPTH deep. The base sheet's own columns win; when two
        lookup sheets bring the same column name, both copies are
        labelled with their sheet ("Doctors Territory", "MRs Territory").
        """

        base_columns = set(base.columns)
        steps = self._plan_steps(sheet, base_columns)

        counts: dict[str, int] = {}
        for step in steps:
            for column in step.columns:
                counts[column] = counts.get(column, 0) + 1

        frame = base.copy()
        sources: dict[str, str] = {}

        for step in steps:
            link = step.link

            if step.parent is None:
                key_column = link.column
            else:
                key_column = steps[step.parent].names[link.column]

            lookup = self.frame(link.lookup_sheet)
            lookup_keys = _as_keys(lookup[link.lookup_column])
            has_key = lookup_keys.notna()
            frame_keys = _as_keys(frame[key_column])

            for column in step.columns:
                name = column if counts[column] == 1 else f"{link.lookup_sheet} {column}"
                mapping = pd.Series(
                    lookup.loc[has_key, column].values,
                    index=lookup_keys[has_key].values,
                )
                frame[name] = frame_keys.map(mapping)
                sources[name] = link.lookup_sheet
                step.names[column] = name

        return EnrichedSheet(sheet=sheet, df=frame, sources=sources)

    def _plan_steps(self, sheet: str, base_columns: set) -> list["_Step"]:
        """Breadth-first lookups from the sheet, each sheet used once."""

        steps: list[_Step] = []
        frontier: list[tuple[str, int | None]] = [(sheet, None)]
        visited = {sheet}

        for _ in range(MAX_LINK_DEPTH):
            next_frontier = []

            for current, parent in frontier:
                for link in self.links():
                    if link.sheet != current or link.lookup_sheet in visited:
                        continue

                    # A deeper hop needs its key to have been added.
                    if parent is not None and link.column not in steps[parent].columns:
                        continue

                    columns = [
                        column
                        for column in self.frame(link.lookup_sheet).columns
                        if column != link.lookup_column and column not in base_columns
                    ]

                    steps.append(_Step(link, parent, columns))
                    visited.add(link.lookup_sheet)
                    next_frontier.append((link.lookup_sheet, len(steps) - 1))

            frontier = next_frontier

        return steps


@dataclass
class _Step:
    link: Link
    parent: int | None
    columns: list[str]
    # Lookup column -> its name in the enriched frame.
    names: dict[str, str] = field(default_factory=dict)


def _normalize(name) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def _as_keys(series: pd.Series) -> pd.Series:
    """Comparable key text: 101, 101.0 and " 101" all become "101"."""

    def key(value):
        if pd.isna(value):
            return None
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        return str(value).strip()

    return series.map(key)
