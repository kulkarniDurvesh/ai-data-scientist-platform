"""
The things to segment, and their features.

    each row         every row is a unit (customers table, one row each)
    per entity       rows grouped by a key (doctor x month -> doctor):
                     numbers averaged, categories by most common value,
                     plus the number of rows when it varies

Features are chosen from column roles: measures, 0/1 columns and
categories with few levels; IDs, dates, free text and constants are left
out with a reason.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from core.schema_inference import DatasetSchema

MAX_LEVELS = 15
RECORDS = "Records"


@dataclass
class Units:
    frame: pd.DataFrame               # one row per unit: [unit column] + features
    unit: str | None                  # entity column, or None for rows
    numeric: list[str]
    categorical: list[str]
    excluded: list[tuple[str, str]] = field(default_factory=list)

    @property
    def features(self) -> list[str]:
        return self.numeric + self.categorical


def unit_options(dataframe: pd.DataFrame, schema: DatasetSchema) -> list[str]:
    """Key columns that repeat: rows can be grouped into one unit per value."""

    return sorted(
        [
            column for column in schema.identifiers
            if 1 < dataframe[column].nunique() < len(dataframe) * 0.9
        ],
        key=lambda column: -dataframe[column].nunique(),
    )


def feature_options(dataframe: pd.DataFrame, schema: DatasetSchema, unit: str | None) -> tuple[list[str], list[tuple[str, str]]]:
    features, excluded = [], []

    for column in dataframe.columns:
        if column == unit:
            continue
        role = schema.role_of(column)
        info = schema.columns[column]
        if role in {"measure"} or (role == "binary" and info.is_numeric):
            features.append(column)
        elif role in {"dimension", "binary"} and info.n_unique <= MAX_LEVELS:
            features.append(column)
        elif role == "dimension":
            excluded.append((column, f"{info.n_unique} categories: too many to profile"))
        elif role == "identifier":
            excluded.append((column, "ID / key"))
        elif role == "time":
            excluded.append((column, "date"))
        elif role == "text":
            excluded.append((column, "free text"))
        else:
            excluded.append((column, "single value"))

    return features, excluded


def build_units(
    dataframe: pd.DataFrame,
    schema: DatasetSchema,
    unit: str | None,
    features: list[str] | None = None,
) -> Units:

    default, excluded = feature_options(dataframe, schema, unit)
    features = [f for f in (features or default) if f in dataframe.columns and f != unit]

    if not features:
        raise ValueError("Choose at least one feature to segment on.")

    numeric = [f for f in features if schema.is_numeric(f)]
    categorical = [f for f in features if f not in numeric]

    if unit is None:
        frame = dataframe[features].copy()
        frame.insert(0, "Row", range(1, len(frame) + 1))
        return Units(frame.reset_index(drop=True), None, numeric, categorical, excluded)

    grouped = dataframe.dropna(subset=[unit]).groupby(unit, sort=True)
    parts = []

    if numeric:
        parts.append(grouped[numeric].mean())
    if categorical:
        parts.append(grouped[categorical].agg(_mode))

    sizes = grouped.size()
    if sizes.nunique() > 1:
        parts.append(sizes.rename(RECORDS).to_frame())
        numeric = numeric + [RECORDS]

    frame = pd.concat(parts, axis=1).reset_index()
    return Units(frame, unit, numeric, categorical, excluded)


def _mode(series: pd.Series):
    values = series.dropna()
    return values.mode().iloc[0] if len(values) else None
