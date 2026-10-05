"""
From a table to regular time series.

    date column + measure (or row count) + optional split column
        -> one series per group (and the total), bucketed by period,
           gaps filled, an incomplete last period dropped

Nothing is specific to a dataset: the measure, aggregation and period
are proposed from column roles and the date range, and can be changed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from core.schema_inference import DatasetSchema
from core.target_analysis import native_grain

ROW_COUNT = "__rows__"
TOTAL = "Total"

# Season length per period: a year of months, weeks, quarters; a week of days.
SEASON = {"D": 7, "W": 52, "M": 12, "Q": 4}
DEFAULT_HORIZON = {"D": 14, "W": 8, "M": 3, "Q": 2}
MAX_GROUPS = 30
# Share of the last period the data must cover to keep that period.
MIN_PERIOD_COVERAGE = 0.8


@dataclass
class SeriesSpec:
    time: str
    measure: str                      # column name or ROW_COUNT
    aggregation: str = "sum"          # sum | mean | count | nunique
    group: str | None = None
    freq: str = "M"
    horizon: int = 3

    @property
    def season(self) -> int:
        return SEASON[self.freq]


@dataclass
class SeriesSet:
    spec: SeriesSpec
    series: dict[str, pd.Series]      # name -> values indexed by period start
    dropped_last: pd.Timestamp | None = None
    notes: list[str] = field(default_factory=list)


def date_options(dataframe: pd.DataFrame) -> list[str]:
    columns = [c for c in dataframe.columns if pd.api.types.is_datetime64_any_dtype(dataframe[c])]
    return sorted(columns, key=lambda c: (dataframe[c].notna().mean(), dataframe[c].nunique()), reverse=True)


def measure_options(
    dataframe: pd.DataFrame,
    schema: DatasetSchema,
    sources: dict[str, str] | None = None,
) -> list[str]:
    """
    Measures recorded on each row, the one with the largest total first
    (usually the main KPI). Columns looked up from other sheets and
    attributes that are constant per key (a doctor's potential, a
    product's unit price) come last: summing them over time means little.
    """

    sources = sources or {}
    measures = [c for c in schema.measures if c in dataframe.columns]
    keys = [
        c for c in dataframe.columns
        if schema.role_of(c) in {"identifier", "dimension"}
        and 1 < dataframe[c].nunique() < len(dataframe) * 0.5
    ]

    def is_attribute(column: str) -> bool:
        if column in sources:
            return True
        for key in keys:
            constant = dataframe.groupby(key)[column].nunique(dropna=True)
            if len(constant) and (constant <= 1).mean() >= 0.95:
                return True
        return False

    totals = {c: float(pd.to_numeric(dataframe[c], errors="coerce").abs().sum()) for c in measures}
    ordered = sorted(
        measures,
        key=lambda c: (not is_attribute(c), schema.columns[c].kind != "ratio", totals[c]),
        reverse=True,
    )
    transactional = [c for c in ordered if not is_attribute(c)]
    return transactional + [ROW_COUNT] + [c for c in ordered if c not in transactional]


def group_options(dataframe: pd.DataFrame, schema: DatasetSchema) -> list[str]:
    return [
        c for c in dataframe.columns
        if schema.role_of(c) in {"dimension", "binary"}
        and 2 <= dataframe[c].nunique() <= MAX_GROUPS
    ]


def default_aggregation(schema: DatasetSchema, measure: str) -> str:
    if measure == ROW_COUNT:
        return "count"
    return "mean" if schema.columns[measure].kind == "ratio" else "sum"


def suggest_freq(dates: pd.Series) -> str:
    """
    Snapshot tables keep their own period (one row per month stays
    monthly). Event dates are bucketed to a business-friendly period:
    months for 18+ months of history, weeks for 4+ months, else days.
    """

    grain = native_grain(dates)
    if grain in {"W", "M", "Q"}:
        return grain
    values = dates.dropna()
    span = (values.max() - values.min()).days if len(values) else 0
    if span >= 540:
        return "M"
    if span >= 120:
        return "W"
    return "D"


def build_series(dataframe: pd.DataFrame, spec: SeriesSpec) -> SeriesSet:
    frame = dataframe.dropna(subset=[spec.time]).copy()
    if frame.empty:
        raise ValueError(f"{spec.time} has no dates.")

    frame["_period"] = frame[spec.time].dt.to_period(spec.freq).dt.start_time
    notes: list[str] = []

    # An incomplete last period would look like a sudden drop: leave it out
    # when the data stops well before the period ends.
    last_period = frame["_period"].max()
    next_period = last_period + pd.tseries.frequencies.to_offset(_offset(spec.freq))
    covered = (frame[spec.time].max().normalize() - last_period).days + 1
    length = (next_period - last_period).days
    dropped = None
    if native_grain(frame[spec.time]) != spec.freq and covered / length < MIN_PERIOD_COVERAGE:
        frame = frame[frame["_period"] < last_period]
        dropped = last_period
        notes.append(
            f"The last period ({last_period:%Y-%m-%d}) is incomplete in the data and was left out."
        )

    groups = {TOTAL: frame}
    if spec.group:
        top = frame[spec.group].value_counts().index[:MAX_GROUPS]
        for value in top:
            groups[str(value)] = frame[frame[spec.group] == value]

    full_index = pd.date_range(frame["_period"].min(), frame["_period"].max(), freq=_offset(spec.freq))
    series = {}

    for name, rows in groups.items():
        values = _aggregate(rows, spec)
        values = values.reindex(full_index)
        if spec.aggregation in {"sum", "count", "nunique"}:
            values = values.fillna(0.0)
        else:
            values = values.interpolate(limit_direction="both")
        series[name] = values.astype(float)

    if len(full_index) < 6:
        raise ValueError(
            f"Only {len(full_index)} periods of history at this frequency: choose a finer period."
        )

    return SeriesSet(spec, series, dropped, notes)


def _aggregate(rows: pd.DataFrame, spec: SeriesSpec) -> pd.Series:
    grouped = rows.groupby("_period")
    if spec.measure == ROW_COUNT or spec.aggregation == "count":
        return grouped.size().astype(float)
    values = pd.to_numeric(rows[spec.measure], errors="coerce")
    grouped = values.groupby(rows["_period"])
    if spec.aggregation == "mean":
        return grouped.mean()
    if spec.aggregation == "nunique":
        return rows.groupby("_period")[spec.measure].nunique().astype(float)
    return grouped.sum()


def _offset(freq: str) -> str:
    return {"D": "D", "W": "W-MON", "M": "MS", "Q": "QS"}[freq]


def future_index(last: pd.Timestamp, freq: str, horizon: int) -> pd.DatetimeIndex:
    return pd.date_range(last, periods=horizon + 1, freq=_offset(freq))[1:]
