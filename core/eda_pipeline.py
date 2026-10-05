"""
End-to-end automatic insight pipeline.

    DataFrame + schema
        -> profile
        -> analytical candidates
        -> statistics
        -> patterns
        -> redundancy filter
        -> evidence + explanations
        -> ranking
        -> insights
"""

from __future__ import annotations

import warnings
from typing import Any

import pandas as pd

from core.evidence import attach_evidence
from core.explanations import attach_explanations
from core.insight import build_insights
from core.insight_filter import filter_redundant_patterns
from core.patterns import detect_pattern
from core.profiler import profile_dataset
from core.ranking import rank_patterns
from core.relationships import generate_candidates
from core.schema_inference import DatasetSchema
from core.statistics import calculate_statistics


def run_insight_pipeline(
    dataframe: pd.DataFrame,
    schema: DatasetSchema,
    top_n: int = 10,
) -> list[dict[str, Any]]:

    profile = profile_dataset(dataframe, schema)

    candidates = generate_candidates(profile)

    patterns = []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        for candidate in candidates:
            try:
                statistics = calculate_statistics(dataframe, candidate)
                pattern = detect_pattern(candidate, statistics)
            except (ValueError, TypeError, KeyError):
                continue

            if pattern is not None:
                patterns.append(pattern)

    if not patterns:
        return []

    patterns = filter_redundant_patterns(patterns, dataframe)
    patterns = attach_evidence(patterns)
    patterns = attach_explanations(patterns)

    ranked = rank_patterns(patterns, top_n=top_n)

    return build_insights(ranked)
