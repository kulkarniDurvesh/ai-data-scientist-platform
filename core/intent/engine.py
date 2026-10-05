"""
The goal box: free text -> validated plan, plus suggestions from the data.

    interpret(text, view, provider)
        1. rules read the text (cues + column names)
        2. if the rules are sure, done; otherwise ask the language model
           (when one is available) and merge: the model maps synonyms,
           the rules' exact column matches fill the model's gaps
        3. `complete` validates every column, fills defaults (assumptions)
           and raises questions for anything it can't resolve

    suggestions(view)   (approach E) goals this data supports, ready to run

Without a language model the box still works: rules plus questions plus
the editable card (approach D).
"""

from __future__ import annotations

from core.llm import LLMError, Provider

from .complete import complete
from .interpret_llm import describe_terms, propose
from .rules import read_goal
from .spec import TASKS, GoalPlan, IntentSpec, Question
from .view import DataView

SUGGESTION_ORDER = ("rank", "recommend", "forecast", "why", "segment", "regress")


def interpret(text: str, view: DataView, provider: Provider | None = None) -> GoalPlan:
    text = (text or "").strip()
    if not text:
        plan = complete(IntentSpec(), view, "rules", text)
        return plan

    reading = read_goal(text, view)

    if reading.confident or provider is None:
        plan = complete(reading.fields, view, "rules", text, reading.soft, reading.sheet_hints)
        _rule_notes(plan, reading, provider is None)
        return plan

    try:
        proposal = propose(text, view, provider)
    except LLMError as error:
        plan = complete(reading.fields, view, "rules", text, reading.soft, reading.sheet_hints)
        plan.llm_error = str(error)
        _rule_notes(plan, reading, True)
        return plan

    merged = {key: value for key, value in proposal.model_dump().items() if value not in (None, [])}
    merged.pop("unknown_terms", None)
    for key, value in reading.fields.items():
        if key == "task" and "task" in merged:
            continue
        merged.setdefault(key, value)
    if merged.get("task") == "ask":
        merged["question"] = text

    proposed = proposal.model_dump()
    soft = {key for key in reading.soft if proposed.get(key) in (None, [])}
    # A model's column that doesn't fit is a guess: fall back to the default
    # (with a note). Only a target or measure that exists nowhere is asked
    # about, since the user named something the data may not have.
    columns = {c for sheet in view.sheets for c in view.native_columns(sheet)}
    for key, value in proposed.items():
        if key in ("task", "unknown_terms") or value in (None, []):
            continue
        if key in ("target", "measure") and isinstance(value, str) and value not in columns:
            continue
        soft.add(key)
    plan = complete(merged, view, "language model", text, soft, reading.sheet_hints)
    if proposal.unknown_terms:
        plan.warnings.append(f"Not matched to the data: {describe_terms(proposal.unknown_terms)}.")
    return plan


def _rule_notes(plan: GoalPlan, reading, no_model: bool) -> None:
    if reading.ambiguous and plan.spec.task is None:
        plan.questions = [q for q in plan.questions if q.field != "task"]
        plan.questions.insert(0, Question(
            "task",
            "Your goal could mean several things. Which one?",
            [t for t in reading.ambiguous if t in plan.options.get("task", [])] or plan.options.get("task", []),
        ))
    if reading.unknown:
        hint = " A language model (Ollama or Azure OpenAI) would map other words for columns." if no_model else ""
        plan.warnings.append(
            f"Not understood: {describe_terms(reading.unknown)} (no column, table or value matches). "
            f"Check the choices below.{hint}"
        )


def suggestions(view: DataView, limit: int = 6) -> list[GoalPlan]:
    """Goals this dataset supports, each validated and ready to run."""

    found: list[GoalPlan] = []
    for task in SUGGESTION_ORDER:
        if not view.sheets_for(task):
            continue
        plan = complete(IntentSpec(task=task), view, "suggestion", "")
        if plan.ready:
            found.append(plan)
        if len(found) >= limit:
            break
    return found


def task_labels() -> dict[str, str]:
    return {name: info["label"] for name, info in TASKS.items()}
