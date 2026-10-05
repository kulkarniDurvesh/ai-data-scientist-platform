"""
Unusual units, with reasons.

    score    Isolation Forest on the prepared features: units that are
             easy to isolate from the rest score high
    flag     the top share (default 1%) by score
    reasons  the features where the unit is furthest from typical, measured
             with robust z-scores (median and MAD, not mean and std, so
             the outliers themselves don't hide), and rare categories
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from core.schema_inference import format_number, humanize

from .cluster import prepare_matrix

RANDOM_STATE = 42
ROBUST_Z = 3.5
RARE_SHARE = 0.02
MAX_REASONS = 3


def anomaly_scores(
    frame: pd.DataFrame,
    numeric: list[str],
    categorical: list[str],
    share: float = 0.01,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Returns (score, flagged, reasons) per unit; higher score = more unusual."""

    matrix = prepare_matrix(frame, numeric, categorical)
    model = IsolationForest(n_estimators=200, random_state=RANDOM_STATE).fit(matrix)
    score = -model.score_samples(matrix)

    count = max(1, int(round(len(frame) * share)))
    threshold = np.sort(score)[-count]
    flagged = score >= threshold

    reasons = _reasons(frame, numeric, categorical)
    return score, flagged, reasons


def _reasons(frame: pd.DataFrame, numeric: list[str], categorical: list[str]) -> list[str]:
    z_scores = {}
    ranges = {}
    for column in numeric:
        values = pd.to_numeric(frame[column], errors="coerce")
        median = values.median()
        mad = (values - median).abs().median()
        scale = 1.4826 * mad if mad > 0 else values.std(ddof=0)
        z_scores[column] = (values - median) / scale if scale and scale > 0 else values * 0
        ranges[column] = (values.quantile(0.25), values.quantile(0.75))

    rare = {}
    for column in categorical:
        shares = frame[column].value_counts(normalize=True)
        rare[column] = set(shares[shares < RARE_SHARE].index)

    texts = []
    for position in range(len(frame)):
        found = []
        for column, z in z_scores.items():
            value = z.iloc[position]
            if pd.notna(value) and abs(value) >= ROBUST_Z:
                low, high = ranges[column]
                found.append((abs(value), f"{humanize(column)} = {format_number(frame[column].iloc[position])} (typical {format_number(low)}–{format_number(high)})"))
        for column, values in rare.items():
            current = frame[column].iloc[position]
            if current in values:
                found.append((ROBUST_Z, f"rare {humanize(column)} = {current}"))
        found.sort(key=lambda item: -item[0])
        texts.append("; ".join(text for _, text in found[:MAX_REASONS]) or "unusual combination of values")

    return texts
