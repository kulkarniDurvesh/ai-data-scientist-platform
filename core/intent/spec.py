"""
A goal stated in words, as a structured, checkable spec.

    "plan which customers each rep should call next month"
        -> IntentSpec(task="recommend", user=<rep key>, item=<customer key>, period="month")

The spec only names columns that exist; anything the user did not say
is filled with the same automatic choices the dashboard tabs make.
`GoalPlan` adds what the platform understood, what it assumed and the
questions it needs answered before running.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any

TASKS: dict[str, dict[str, str]] = {
    "rank": {
        "label": "Rank / prioritise",
        "description": "Order rows or entities by how likely a yes/no outcome is.",
        "tab": "model",
    },
    "classify": {
        "label": "Predict a yes/no outcome",
        "description": "Estimate the probability of a two-valued column.",
        "tab": "model",
    },
    "regress": {
        "label": "Predict a number",
        "description": "Estimate a numeric column for each row.",
        "tab": "model",
    },
    "recommend": {
        "label": "Recommend / plan contacts",
        "description": "Decide which items each user should contact next, as a day-by-day plan.",
        "tab": "recommend",
    },
    "forecast": {
        "label": "Forecast over time",
        "description": "Project a measure (or a count of rows) for the next periods.",
        "tab": "forecast",
    },
    "segment": {
        "label": "Find segments and anomalies",
        "description": "Group similar units and flag unusual ones.",
        "tab": "segments",
    },
    "why": {
        "label": "Explain a change",
        "description": "Find what drove the change of a measure between two periods.",
        "tab": "why",
    },
    "ask": {
        "label": "Answer a question",
        "description": "Look something up in the data (lists, counts, totals, charts).",
        "tab": "ask",
    },
}

MODEL_TASKS = ("rank", "classify", "regress")


@dataclass
class IntentSpec:
    task: str | None = None
    sheet: str | None = None
    # models
    target: str | None = None
    # forecast and why
    time: str | None = None
    measure: str | None = None
    aggregation: str | None = None
    group: str | None = None
    freq: str | None = None
    horizon: int | None = None
    compare: str | None = None
    dimensions: list[str] | None = None
    attention: str | None = None
    # recommend
    user: str | None = None
    item: str | None = None
    outcome: str | None = None
    period: str | None = None
    days: int | None = None
    # segments
    unit: str | None = None
    features: list[str] | None = None
    # ask
    question: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "IntentSpec":
        names = {f.name for f in fields(cls)}
        return cls(**{key: value for key, value in (data or {}).items() if key in names})


@dataclass
class Question:
    """Something the platform needs to know before it can run."""

    field: str
    text: str
    options: list[Any] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GoalPlan:
    spec: IntentSpec
    text: str = ""
    source: str = "rules"                       # rules | language model | suggestion | edited
    title: str = ""
    summary: str = ""
    assumptions: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    questions: list[Question] = field(default_factory=list)
    options: dict[str, list[Any]] = field(default_factory=dict)
    llm_error: str | None = None

    @property
    def ready(self) -> bool:
        return self.spec.task is not None and not self.questions

    @property
    def tab(self) -> str | None:
        return TASKS[self.spec.task]["tab"] if self.spec.task in TASKS else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec": self.spec.to_dict(),
            "text": self.text,
            "source": self.source,
            "title": self.title,
            "summary": self.summary,
            "ready": self.ready,
            "tab": self.tab,
            "assumptions": list(self.assumptions),
            "warnings": list(self.warnings),
            "questions": [q.to_dict() for q in self.questions],
            "options": {key: list(values) for key, values in self.options.items()},
            "llm_error": self.llm_error,
        }
