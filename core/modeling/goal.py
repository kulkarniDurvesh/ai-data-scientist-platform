"""
What the user wants to build, as a structured goal.

    goal type (classify / regress / rank) + target
        -> GoalSpec: features, excluded columns (with reasons), time column,
           entity, positive class

The GoalSpec is the contract between whoever states the goal (the form
today, an LLM agent later) and the model pipelines. It is proposed from
the Phase 2 target analysis, so leakage exclusions, the snapshot time
column and panel structure are reused, not recomputed differently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from core.schema_inference import DatasetSchema
from core.target_analysis import TargetReport, analyze_target, detect_targets


GOAL_TYPES = {
    "classify": {
        "label": "Predict a yes/no outcome",
        "description": "Estimate the probability of an event, e.g. will a "
        "customer buy again next month, will an account churn.",
        "target_roles": {"binary"},
    },
    "rank": {
        "label": "Rank / prioritise",
        "description": "Order entities by how likely the outcome is, e.g. which "
        "accounts to contact first. Same model as yes/no, judged by how well "
        "the top of the list captures the outcome.",
        "target_roles": {"binary"},
    },
    "regress": {
        "label": "Predict a number",
        "description": "Estimate a quantity, e.g. units ordered, revenue, cost.",
        "target_roles": {"measure"},
    },
}

# Categories with more distinct values than this are left out of a first
# model: one-hot encoding them adds many sparse columns for little gain.
MAX_CATEGORIES = 50


@dataclass
class GoalSpec:
    goal_type: str
    target: str
    features: list[str]
    excluded: list[tuple[str, str]] = field(default_factory=list)
    time_column: str | None = None
    entity: str | None = None
    positive_label: Any = None

    @property
    def task(self) -> str:
        return "regression" if self.goal_type == "regress" else "classification"

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal_type": self.goal_type,
            "target": self.target,
            "features": list(self.features),
            "excluded": [list(item) for item in self.excluded],
            "time_column": self.time_column,
            "entity": self.entity,
            "positive_label": None if self.positive_label is None else str(self.positive_label),
        }


def target_options(
    dataframe: pd.DataFrame,
    schema: DatasetSchema,
    goal_type: str,
) -> list[str]:
    """Columns that fit the goal type, the most likely target first."""

    roles = GOAL_TYPES[goal_type]["target_roles"]
    suggested = [
        candidate.column
        for candidate in detect_targets(dataframe, schema)
        if schema.role_of(candidate.column) in roles
    ]
    others = [
        column
        for column in dataframe.columns
        if schema.role_of(column) in roles and column not in suggested
    ]
    return suggested + others


def propose_goal(
    dataframe: pd.DataFrame,
    schema: DatasetSchema,
    goal_type: str,
    target: str,
    report: TargetReport | None = None,
) -> GoalSpec:
    """Features, exclusions and split settings for a goal on this dataset."""

    if goal_type not in GOAL_TYPES:
        raise ValueError(f"Unknown goal type '{goal_type}'.")

    roles = GOAL_TYPES[goal_type]["target_roles"]
    if schema.role_of(target) not in roles:
        kind = "two-valued" if "binary" in roles else "numeric"
        raise ValueError(
            f"'{target}' doesn't fit this goal: choose a {kind} column."
        )

    report = report or analyze_target(dataframe, schema, target)

    excluded: dict[str, str] = {column: reason for column, reason in report.excluded}

    for finding in report.findings:
        if (
            finding.severity == "critical"
            and finding.column
            and finding.column != target
            and finding.column not in excluded
        ):
            excluded[finding.column] = f"possible leakage: {finding.title.split(': ', 1)[-1]}"

    for column in dataframe.columns:
        if column == target or column in excluded:
            continue
        info = schema.columns[column]
        if info.role == "dimension" and info.n_unique > MAX_CATEGORIES:
            excluded[column] = f"{info.n_unique} categories: too many for a first model"

    features = [
        column
        for column in dataframe.columns
        if column != target and column not in excluded
    ]

    if not features:
        raise ValueError("No usable feature columns are left after exclusions.")

    time_column = report.time_column if report.split and report.split.method == "time" else None

    return GoalSpec(
        goal_type=goal_type,
        target=target,
        features=features,
        excluded=list(excluded.items()),
        time_column=time_column,
        entity=report.panel.entity if report.panel else _entity_column(schema, target),
        positive_label=report.positive_label,
    )


def _entity_column(schema: DatasetSchema, target: str) -> str | None:
    """The first key column, to name rows in scored output."""

    for column in schema.identifiers:
        if column != target:
            return column
    return None
