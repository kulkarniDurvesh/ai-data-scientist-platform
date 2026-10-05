"""
Language-model reading of a goal (approach C).

The model sees the task catalogue and a compact profile of the tables
(column names, roles, a few sample values) and returns a structured
proposal. It never sees the full data and never computes anything; its
proposal is validated by `complete` like any other.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from core.llm import Provider, structured
from core.schema_inference import humanize

from .spec import TASKS
from .view import DataView

MAX_COLUMNS_PER_SHEET = 40
SAMPLES = 3

TaskName = Literal["rank", "classify", "regress", "recommend", "forecast", "segment", "why", "ask"]


class GoalProposal(BaseModel):
    task: TaskName | None = Field(None, description="The one task that best fits the goal")
    sheet: str | None = Field(None, description="Table to use, exactly as listed; null if unsure")
    target: str | None = Field(None, description="rank/classify/regress: column to predict")
    measure: str | None = Field(None, description="forecast/why: numeric column (or a key column to count rows)")
    time: str | None = Field(None, description="Date column, only if the user names one")
    group: str | None = Field(None, description="forecast: column to split the series by")
    user: str | None = Field(None, description="recommend: key column of who acts")
    item: str | None = Field(None, description="recommend: key column of what they act on")
    unit: str | None = Field(None, description="segment: key column, one unit per value")
    attention: str | None = Field(None, description="why: entity column to rank")
    features: list[str] | None = Field(None, description="segment: columns to use, only if named")
    dimensions: list[str] | None = Field(None, description="why: columns to drill into, only if named")
    freq: Literal["D", "W", "M", "Q"] | None = None
    horizon: int | None = Field(None, description="forecast: number of future periods")
    period: Literal["days", "month"] | None = Field(None, description="recommend: next month, or a number of days")
    days: int | None = None
    compare: Literal["previous", "year"] | None = None
    unknown_terms: list[str] = Field(default_factory=list, description="Words of the goal that match no column or value")


SYSTEM = """You turn a user's goal for a dataset into a structured request.
Rules:
- Choose exactly one task from the catalogue.
- Use column and table names exactly as listed. Never invent names.
- The user may use other words for a column (synonyms, plurals, abbreviations): map them to the listed column.
- Fill only fields the user states or clearly implies; leave the rest null (the system fills defaults).
- Put words that match no column, table or value in unknown_terms.
- Do not compute or estimate any numbers from the data.
Reply with JSON only."""


def catalogue(view: DataView) -> str:
    lines = ["Tasks:"]
    for name, info in TASKS.items():
        lines.append(f"- {name}: {info['description']}")
    lines.append("")
    lines.append("Tables:")
    for sheet in view.sheets:
        try:
            frame, schema, _, _ = view.context(sheet)
        except (ValueError, KeyError, TypeError):
            continue
        lines.append(f'## "{view.label(sheet)}" ({len(frame)} rows)')
        for column in view.native_columns(sheet)[:MAX_COLUMNS_PER_SHEET]:
            samples = frame[column].dropna().astype(str).unique()[:SAMPLES]
            shown = ", ".join(value[:20] for value in samples)
            lines.append(f"- {column} [{schema.role_of(column)}] e.g. {shown}")
    return "\n".join(lines)


def propose(text: str, view: DataView, provider: Provider) -> GoalProposal:
    user = f"{catalogue(view)}\n\nGoal: {text.strip()}"
    proposal = structured(provider, SYSTEM, user, GoalProposal)
    # Table names are shown with the dataset name for single-table data.
    if proposal.sheet is not None and proposal.sheet not in view.sheets:
        proposal.sheet = next((s for s in view.sheets if view.label(s).lower() == proposal.sheet.lower()), None)
    return proposal


def describe_terms(terms: list[str]) -> str:
    return ", ".join(f"'{t}'" for t in terms)
