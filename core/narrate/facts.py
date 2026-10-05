"""
Fact sheets: the computed numbers a narrative may use, already formatted.

Every narrative, written by a template or a language model, is built
from these facts only. Each fact keeps its display text so the numbers
in a narrative can be checked against them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from core.schema_inference import DatasetSchema, format_number, humanize

ROLE_WORDS = {
    "measure": ("numeric measure", "numeric measures"),
    "dimension": ("category", "categories"),
    "binary": ("yes/no column", "yes/no columns"),
    "identifier": ("key column", "key columns"),
    "time": ("date column", "date columns"),
    "text": ("free-text column", "free-text columns"),
}


@dataclass
class Fact:
    key: str
    text: str


@dataclass
class FactSheet:
    kind: str                     # dataset | model
    title: str
    facts: list[Fact] = field(default_factory=list)

    def add(self, key: str, text: str) -> None:
        self.facts.append(Fact(key, text))

    def get(self, key: str) -> str | None:
        return next((f.text for f in self.facts if f.key == key), None)

    def lines(self) -> str:
        return "\n".join(f"- {f.key}: {f.text}" for f in self.facts)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "title": self.title, "facts": {f.key: f.text for f in self.facts}}


def dataset_facts(
    name: str,
    frame: pd.DataFrame,
    schema: DatasetSchema,
    quality: list[dict[str, Any]],
    insights: list[dict[str, Any]],
    sheets: list[str] | None = None,
) -> FactSheet:
    sheet = FactSheet("dataset", name)
    sheet.add("rows", format_number(len(frame)))
    sheet.add("columns", format_number(frame.shape[1]))
    if sheets and len(sheets) > 1:
        sheet.add("tables in the file", format_number(len(sheets)))

    roles = pd.Series([c.role for c in schema.columns.values()]).value_counts()
    sheet.add("column roles", ", ".join(
        f"{count} {ROLE_WORDS[role][count != 1]}" for role, count in roles.items() if role in ROLE_WORDS
    ))

    time = schema.primary_time
    if time and time in frame.columns:
        dates = pd.to_datetime(frame[time], errors="coerce").dropna()
        if not dates.empty:
            sheet.add("time axis", humanize(time))
            sheet.add("time span", f"{dates.min():%b %Y} to {dates.max():%b %Y}")

    missing = float(frame.isna().mean().mean()) if frame.size else 0.0
    sheet.add("missing cells", f"{missing:.1%}")

    serious = [i for i in quality if i.get("severity") in ("critical", "warning")]
    sheet.add("quality findings", format_number(len(quality)))
    sheet.add("findings needing attention", format_number(len(serious)))
    for index, issue in enumerate(serious[:2], 1):
        sheet.add(f"quality issue {index}", str(issue.get("message", "")))

    for index, insight in enumerate(insights[:3], 1):
        text = insight.get("description") or insight.get("explanation")
        if text:
            sheet.add(f"insight {index}", str(text))
    return sheet


def model_facts(result) -> FactSheet:
    spec = result.spec
    sheet = FactSheet("model", f"{spec.goal_type} {spec.target}")
    goal = {"rank": "rank rows by how likely", "classify": "predict whether", "regress": "predict"}[spec.goal_type]
    sheet.add("goal", f"{goal} {humanize(spec.target)}")
    sheet.add("chosen model", result.best_name)
    sheet.add("test period", result.split.description)

    best, base = result.test_metrics, result.baseline_metrics
    if result.task == "classification":
        share = best.get("top_share", 0.2)
        sheet.add("positive rate", f"{best.get('positive_rate', 0):.1%}")
        sheet.add("PR-AUC", f"{best.get('pr_auc', float('nan')):.3f} (baseline {base.get('pr_auc', float('nan')):.3f})")
        sheet.add("ROC-AUC", f"{best.get('roc_auc', float('nan')):.3f} (0.5 is random)")
        sheet.add("top of the list", f"the top {share:.0%} captures {best.get('recall_at_top', 0):.0%} of all positives")
        sheet.add("lift at the top", f"{best.get('lift_at_top', 0):.1f}x the average rate")
    else:
        mae, base_mae = best.get("mae"), base.get("mae")
        sheet.add("MAE", f"{format_number(mae)} (baseline {format_number(base_mae)})")
        if base_mae:
            sheet.add("error reduction", f"{1 - mae / base_mae:.0%} lower error than the baseline")
        if best.get("r2") == best.get("r2"):
            sheet.add("R2", f"{best['r2']:.2f}")

    importance = result.importance
    if importance is not None and not importance.empty:
        top = importance[importance["Importance"] > 0].head(3)["Feature"].tolist()
        if top:
            sheet.add("strongest features", ", ".join(humanize(f) for f in top))
    for index, warning in enumerate(result.warnings[:2], 1):
        sheet.add(f"warning {index}", warning)
    return sheet
