"""
Segments, anomalies and correlations in one run.

    table + SegmentSpec
        -> units (rows, or one per entity) and features
        -> correlation report
        -> segments (k-means with silhouette, or bands of a measure)
        -> anomaly scores with reasons
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from core.schema_inference import DatasetSchema, humanize

from .anomaly import anomaly_scores
from .cluster import Clustering, band_segments, kmeans_segments
from .correlation import CorrelationReport, correlation_report
from .units import Units, build_units

Progress = Callable[[str], None]


@dataclass
class SegmentSpec:
    unit: str | None = None
    features: list[str] | None = None
    method: str = "kmeans"            # kmeans | bands
    k: int | None = None
    band_measure: str | None = None
    anomaly_share: float = 0.01


@dataclass
class SegmentResult:
    spec: SegmentSpec
    units: Units
    clustering: Clustering
    table: pd.DataFrame               # units with Segment, anomaly score, flag, reasons
    anomalies: pd.DataFrame
    correlation: CorrelationReport
    notes: list[str] = field(default_factory=list)
    summary: str = ""
    seconds: float = 0.0


def build_segments(
    dataframe: pd.DataFrame,
    schema: DatasetSchema,
    spec: SegmentSpec,
    progress: Progress | None = None,
) -> SegmentResult:

    started = time.time()
    say = progress or (lambda message: None)

    say("Preparing units and features")
    units = build_units(dataframe, schema, spec.unit, spec.features)
    frame = units.frame

    say("Measuring correlations")
    correlation = correlation_report(frame, units.numeric, units.categorical)

    say("Finding segments")
    if spec.method == "bands":
        measure = spec.band_measure or (units.numeric[0] if units.numeric else None)
        if measure is None or measure not in frame.columns:
            raise ValueError("Bands need a numeric measure.")
        clustering = band_segments(frame, measure, units.numeric, units.categorical)
    else:
        clustering = kmeans_segments(frame, units.numeric, units.categorical, spec.k)

    say("Scoring anomalies")
    score, flagged, reasons = anomaly_scores(frame, units.numeric, units.categorical, spec.anomaly_share)

    table = frame.copy()
    table.insert(1, "Segment", [clustering.names[label] for label in clustering.labels])
    table["Anomaly score"] = np.round(score, 3)
    table["Unusual"] = np.where(flagged, "yes", "")
    table["Why unusual"] = np.where(flagged, reasons, "")

    anomalies = (
        table[flagged]
        .sort_values("Anomaly score", ascending=False)
        .drop(columns=["Unusual"])
        .reset_index(drop=True)
    )

    notes = []
    if clustering.silhouette is not None and clustering.silhouette < 0.15:
        notes.append(
            f"Silhouette {clustering.silhouette:.2f}: the segments overlap a lot, so treat "
            f"them as broad tendencies rather than sharp groups."
        )
    if len(correlation.redundant):
        notes.append(
            f"{len(correlation.redundant)} pair(s) of features are almost redundant "
            f"(|rank correlation| ≥ 0.9); see the correlation section."
        )

    result = SegmentResult(
        spec=spec, units=units, clustering=clustering, table=table,
        anomalies=anomalies, correlation=correlation, notes=notes,
        seconds=round(time.time() - started, 1),
    )
    result.summary = _summarize(result)
    say("Done")
    return result


def _summarize(result: SegmentResult) -> str:
    clustering = result.clustering
    unit = humanize(result.units.unit) if result.units.unit else "row"
    profiles = clustering.profiles.sort_values("Units", ascending=False)

    parts = [f"{row.Segment} ({row.Share:.0%})" for row in profiles.itertuples()]
    text = (
        f"{len(result.table):,} units (one per {unit}) on {len(result.units.features)} features. "
        f"{clustering.method} found {clustering.k} segments: " + "; ".join(parts) + "."
    )
    if clustering.silhouette is not None:
        text += f" Silhouette {clustering.silhouette:.2f} (−1 to 1; higher = better separated)."
    text += f" {len(result.anomalies)} unusual unit(s) flagged."
    return text
