"""
Validate a proposed goal against the data and fill in what is missing.

Whoever proposed the spec (rules, a language model, a suggestion, a user
editing the card), every column is checked against the real tables and
the role the task needs; unknown names are corrected when the match is
unambiguous, otherwise they become questions. Fields left empty get the
same defaults as the dashboard tabs, listed as assumptions so the user
can see and change them.
"""

from __future__ import annotations

import difflib
from typing import Any

from core.forecast import DEFAULT_HORIZON, ROW_COUNT, date_options, default_aggregation, measure_options, suggest_freq
from core.forecast import group_options as series_group_options
from core.modeling import target_options
from core.recommend import complete_roles, default_roles, role_options
from core.schema_inference import humanize, key_entity_name, name_tokens, same_word
from core.segment import feature_options, unit_options
from core.why import attention_options, drill_options

from .spec import MODEL_TASKS, TASKS, GoalPlan, IntentSpec, Question
from .view import DataView

COLUMN_FIELDS = ("target", "time", "measure", "group", "attention", "user", "item", "outcome", "unit")
LIST_FIELDS = ("dimensions", "features")
FREQ_WORDS = {"D": ("day", "days"), "W": ("week", "weeks"), "M": ("month", "months"), "Q": ("quarter", "quarters")}
MAX_OPTIONS = 40


def complete(
    raw: IntentSpec | dict | None,
    view: DataView,
    source: str = "rules",
    text: str = "",
    soft: set[str] | tuple = (),
    sheet_hints: list | tuple = (),
) -> GoalPlan:
    """
    `soft` fields are guesses (partial name matches): when they don't fit
    they fall back to the default instead of raising a question.
    `sheet_hints` are tables named in the text, preferred on a tie.
    """

    spec = raw if isinstance(raw, IntentSpec) else IntentSpec.from_dict(raw)
    spec = IntentSpec.from_dict(spec.to_dict())
    plan = GoalPlan(spec=spec, text=text, source=source)

    feasible = [task for task in TASKS if view.sheets_for(task)]
    plan.options["task"] = feasible

    if spec.task not in TASKS:
        spec.task = None
        plan.questions.append(Question("task", "What would you like to do with this data?", feasible))
        plan.summary = "The goal is not clear yet: choose what you want to do."
        return plan
    if spec.task not in feasible:
        plan.warnings.append(_infeasible(spec.task))
        plan.questions.append(Question("task", "This data can't support that goal. Choose another one:", feasible))
        return plan

    _choose_sheet(plan, view, list(sheet_hints))
    if any(q.field == "sheet" for q in plan.questions):
        return plan

    handler = {
        "rank": _model, "classify": _model, "regress": _model,
        "forecast": _series, "why": _why, "segment": _segment,
        "recommend": _recommend, "ask": _ask,
    }[spec.task]
    plan.options["_all"] = list(view.context(spec.sheet)[0].columns)
    plan.options["_soft"] = list(soft)
    try:
        handler(plan, view)
    except (ValueError, KeyError, TypeError) as error:
        plan.questions.append(Question("task", f"This goal can't run as set: {error}", feasible))
    plan.options.pop("_all", None)
    plan.options.pop("_soft", None)

    plan.summary = describe(plan.spec, view)
    plan.title = plan.title or TASKS[spec.task]["label"]
    return plan


# ----------------------------------------------------------------------
# Shared helpers
# ----------------------------------------------------------------------

def match_column(name: Any, allowed: list[str]) -> str | None:
    """The allowed column a proposed name refers to, if the match is clear."""

    if name is None or not allowed:
        return None
    name = str(name).strip()
    if name in allowed:
        return name
    lowered = {c.lower(): c for c in allowed}
    if name.lower() in lowered:
        return lowered[name.lower()]

    tokens = name_tokens(name)
    by_tokens = [c for c in allowed if _same_tokens(tokens, name_tokens(c))]
    if len(by_tokens) == 1:
        return by_tokens[0]

    close = difflib.get_close_matches(name.lower(), list(lowered), n=2, cutoff=0.85)
    if len(close) == 1:
        return lowered[close[0]]
    return None


def _same_tokens(first: list[str], second: list[str]) -> bool:
    return len(first) == len(second) and all(same_word(a, b) for a, b in zip(first, second))


def _pick(plan: GoalPlan, field: str, allowed: list[str], default: Any, label: str, required: bool = True) -> Any:
    """Validate one column field; fall back to the default with a note."""

    spec = plan.spec
    proposed = getattr(spec, field)
    plan.options[field] = list(allowed)[:MAX_OPTIONS]

    if proposed is not None:
        chosen = match_column(proposed, allowed)
        if chosen is not None:
            if chosen != proposed:
                plan.assumptions.append(f"Read '{proposed}' as the column '{chosen}'.")
            setattr(spec, field, chosen)
            return chosen
        setattr(spec, field, None)
        if field in plan.options.get("_soft", []):
            plan.assumptions.append(f"Ignored '{proposed}' as the {label} (it doesn't fit).")
            return _pick(plan, field, allowed, default, label, required)
        reason = "doesn't fit here" if proposed in plan.options.get("_all", []) else "isn't a column of this table"
        plan.questions.append(Question(field, f"'{proposed}' {reason}. Which column should be the {label}?", list(allowed)[:MAX_OPTIONS]))
        setattr(spec, field, None)
        return None

    if default is None:
        if required:
            plan.questions.append(Question(field, f"Which column should be the {label}?", list(allowed)[:MAX_OPTIONS]))
        return None

    setattr(spec, field, default)
    plan.assumptions.append(f"{label.capitalize()}: {_name(default)} (chosen automatically).")
    return default


def _pick_list(plan: GoalPlan, field: str, allowed: list[str], label: str) -> list[str] | None:
    proposed = getattr(plan.spec, field)
    plan.options[field] = list(allowed)[:MAX_OPTIONS]
    if not proposed:
        setattr(plan.spec, field, None)
        plan.assumptions.append(f"{label.capitalize()}: all suitable columns (chosen automatically).")
        return None
    chosen, dropped = [], []
    for name in proposed:
        column = match_column(name, allowed)
        (chosen if column else dropped).append(column or name)
    if dropped:
        plan.warnings.append(f"Not usable as {label}: {', '.join(map(str, dropped))}.")
    setattr(plan.spec, field, list(dict.fromkeys(chosen)) or None)
    return getattr(plan.spec, field)


def noun(column: str | None, plural: bool = False) -> str:
    """The thing a key column names, keeping acronyms: 'MRId' -> 'MR', 'CustomerId' -> 'customer'."""

    import re

    base = key_entity_name(column) or humanize(column).lower()
    words = [
        w.upper() if len(w) <= 4 and re.search(re.escape(w.upper()) + r"(?![a-z])", str(column)) else w
        for w in base.split()
    ]
    text = " ".join(words)
    if plural:
        if text.endswith("y") and not text.endswith(("ay", "ey", "oy", "uy")):
            text = text[:-1] + "ies"
        elif not text.endswith("s"):
            text += "s"
    return text


def _name(value: Any) -> str:
    return "number of rows" if value == ROW_COUNT else f"'{value}'"


def _infeasible(task: str) -> str:
    return {
        "recommend": "Recommendations need an interaction table: two repeated key columns (who and what), a date and an outcome.",
        "forecast": "Forecasting needs a table with a date column and at least 20 rows.",
        "why": "Explaining a change needs a table with a date column and at least 20 rows.",
        "segment": "Segmenting needs a table with numeric or category columns.",
    }.get(task, "This data can't support that goal.")


def _choose_sheet(plan: GoalPlan, view: DataView, hints: list) -> None:
    spec = plan.spec
    candidates = view.sheets_for(spec.task)
    plan.options["sheet"] = list(candidates)

    if spec.sheet is not None and spec.sheet not in view.sheets:
        plan.warnings.append(f"There is no table named '{spec.sheet}'.")
        spec.sheet = None

    if spec.sheet is not None and spec.sheet in candidates:
        return
    if spec.sheet is not None and spec.task in MODEL_TASKS:
        plan.warnings.append(
            f"Model goals use the loaded table '{view.label(view.model_sheet)}'; "
            f"load '{spec.sheet}' instead to model it."
        )

    named = [getattr(spec, f) for f in COLUMN_FIELDS if getattr(spec, f)] + [
        c for f in LIST_FIELDS for c in (getattr(spec, f) or [])
    ]
    hinted = [s for s in candidates if s in hints]
    if len(candidates) == 1:
        spec.sheet = candidates[0]
    elif not named:
        spec.sheet = hinted[0] if hinted else candidates[0]
    else:
        def coverage(sheet):
            frame = view.context(sheet)[0]
            native = set(view.native_columns(sheet))
            return (sum(n in native for n in named), sheet in hints, sum(n in frame.columns for n in named))

        spec.sheet = max(candidates, key=lambda sheet: (coverage(sheet), -candidates.index(sheet)))

    if len(view.sheets) > 1:
        plan.assumptions.append(f"Table: '{view.label(spec.sheet)}'.")

    if spec.task in MODEL_TASKS and spec.target:
        frame = view.context(spec.sheet)[0]
        if spec.target not in frame.columns:
            home = next((s for s in view.sheets if spec.target in view.native_columns(s)), None)
            if home is not None:
                plan.warnings.append(
                    f"'{spec.target}' is in the table '{view.label(home)}'; model goals use the loaded "
                    f"table '{view.label(view.model_sheet)}' (load '{view.label(home)}' to model it)."
                )


# ----------------------------------------------------------------------
# Tasks
# ----------------------------------------------------------------------

def _model(plan: GoalPlan, view: DataView) -> None:
    spec = plan.spec
    frame, schema, _, _ = view.context(spec.sheet)
    options = target_options(frame, schema, spec.task)
    if not options:
        kind = "numeric" if spec.task == "regress" else "two-valued (yes/no)"
        raise ValueError(f"the table has no {kind} column to predict")
    _pick(plan, "target", options, options[0], "column to predict")
    if spec.target:
        entity = next((c for c in schema.identifiers if key_entity_name(c)), None)
        verb = {"rank": "Rank by", "classify": "Predict", "regress": "Predict"}[spec.task]
        plan.title = f"{verb} {humanize(spec.target)}" + (f" for each {noun(entity)}" if entity else "")


def _measures(view: DataView, sheet) -> list[str]:
    frame, schema, sources, _ = view.context(sheet)
    return measure_options(frame, schema, sources)


def _measure_and_time(plan: GoalPlan, view: DataView) -> None:
    spec = plan.spec
    frame, schema, _, _ = view.context(spec.sheet)
    dates = date_options(frame)
    measures = _measures(view, spec.sheet)

    # A key column named as the measure means "how many": count rows.
    if spec.measure and spec.measure not in measures and spec.measure in frame.columns and schema.role_of(spec.measure) == "identifier":
        plan.assumptions.append(f"Counting rows (one per '{spec.measure}').")
        spec.measure = ROW_COUNT

    _pick(plan, "time", dates, dates[0] if dates else None, "date column")
    _pick(plan, "measure", measures, measures[0] if measures else None, "measure")
    if spec.measure and spec.measure != ROW_COUNT and spec.aggregation not in ("sum", "mean", "count", "nunique"):
        spec.aggregation = default_aggregation(schema, spec.measure)
    elif spec.measure == ROW_COUNT:
        spec.aggregation = "count"
    if spec.time and spec.freq not in FREQ_WORDS:
        spec.freq = suggest_freq(frame[spec.time])
    plan.options["freq"] = list(FREQ_WORDS)


def _series(plan: GoalPlan, view: DataView) -> None:
    spec = plan.spec
    _measure_and_time(plan, view)
    frame, schema, _, _ = view.context(spec.sheet)
    groups = series_group_options(frame, schema)
    if spec.group is not None:
        _pick(plan, "group", groups, None, "column to split by", required=False)
    plan.options["group"] = groups[:MAX_OPTIONS]
    if spec.freq and not spec.horizon:
        spec.horizon = DEFAULT_HORIZON[spec.freq]
    spec.horizon = max(1, min(int(spec.horizon or 1), 60))
    if spec.measure and spec.time:
        plan.title = f"Forecast {_measure_label(spec)}"


def _why(plan: GoalPlan, view: DataView) -> None:
    spec = plan.spec
    _measure_and_time(plan, view)
    if spec.aggregation == "nunique":
        spec.aggregation = "sum"
    frame, schema, _, _ = view.context(spec.sheet)
    _pick_list(plan, "dimensions", drill_options(frame, schema), "columns to drill into")
    entities = attention_options(frame, schema)
    _pick(plan, "attention", entities, entities[0] if entities else None, "entity to rank", required=False)
    spec.compare = spec.compare if spec.compare in ("previous", "year") else "previous"
    if spec.measure:
        plan.title = f"Explain the change in {_measure_label(spec)}"


def _segment(plan: GoalPlan, view: DataView) -> None:
    spec = plan.spec
    frame, schema, _, _ = view.context(spec.sheet)
    units = unit_options(frame, schema)
    default_unit = units[0] if units and len(frame) / max(frame[units[0]].nunique(), 1) >= 2 else None
    # A key with one row per value: each row already is one unit.
    if spec.unit in frame.columns and spec.unit not in units and frame[spec.unit].nunique() >= 0.9 * len(frame):
        plan.assumptions.append(f"Each row is one '{spec.unit}'.")
        plan.title = f"Find groups of similar {noun(spec.unit, plural=True)}"
        spec.unit = None
        default_unit = None
        named_rows = True
    else:
        named_rows = False
    _pick(plan, "unit", units, default_unit, "unit to segment", required=False)
    if spec.unit is None and not named_rows:
        plan.assumptions.append("Each row is one unit.")
    features, _ = feature_options(frame, schema, spec.unit)
    _pick_list(plan, "features", features, "features")
    if not named_rows:
        plan.title = f"Find groups of similar {noun(spec.unit, plural=True) if spec.unit else 'rows'}"


def _recommend(plan: GoalPlan, view: DataView) -> None:
    spec = plan.spec
    frame, schema, sources, links = view.context(spec.sheet)
    options = role_options(frame, schema, sources)
    defaults = default_roles(frame, schema, sources, links)

    keys = list(dict.fromkeys(options.items + options.users))
    # Only one key named: put it in the role it has by default.
    if spec.user and not spec.item and match_column(spec.user, keys) == defaults.item:
        spec.item, spec.user = spec.user, None

    _pick(plan, "user", keys, defaults.user, "who acts (user)")
    remaining = [k for k in keys if k != spec.user] or keys
    item_default = defaults.item if defaults.item != spec.user else (remaining[0] if remaining else None)
    _pick(plan, "item", remaining, item_default, "what they act on (item)")
    _pick(plan, "time", options.times, defaults.time, "date column")
    _pick(plan, "outcome", options.outcomes, defaults.outcome, "outcome column")

    if spec.user and spec.item and spec.time and spec.outcome:
        roles = complete_roles(frame, schema, sources, links, user=spec.user, item=spec.item, time=spec.time, outcome=spec.outcome)
        plan.assumptions.append(
            "Success means " + _name(spec.outcome) + " is " + " or ".join(f"'{v}'" for v in roles.success_values) + "."
        )
    spec.period = spec.period if spec.period in ("days", "month") else "month"
    spec.days = int(spec.days or 5)
    plan.options["period"] = ["month", "days"]
    when = "next month, day by day" if spec.period == "month" else f"the next {spec.days} days"
    plan.title = f"Plan which {noun(spec.item, plural=True)} each {noun(spec.user)} should contact"
    plan.assumptions.append(f"Plan: {when}.")


def _ask(plan: GoalPlan, view: DataView) -> None:
    spec = plan.spec
    spec.question = (spec.question or plan.text or "").strip()
    if not spec.question:
        plan.questions.append(Question("question", "What is your question?"))
    plan.title = "Answer a question"


# ----------------------------------------------------------------------
# Description
# ----------------------------------------------------------------------

def _measure_label(spec: IntentSpec) -> str:
    if spec.measure == ROW_COUNT:
        return "the number of rows"
    word = {"sum": "total", "mean": "average", "nunique": "distinct", "count": "count of"}.get(spec.aggregation or "sum", "total")
    label = humanize(spec.measure)
    return label if label.lower().startswith(word) else f"{word} {label}"


def describe(spec: IntentSpec, view: DataView) -> str:
    """One sentence saying what will run, in the data's own words."""

    table = f" (table '{view.label(spec.sheet)}')" if len(view.sheets) > 1 and spec.sheet is not None else ""
    task = spec.task
    if task in MODEL_TASKS and spec.target:
        verb = {"rank": "Rank rows by how likely", "classify": "Predict whether", "regress": "Predict"}[task]
        tail = " is positive" if task != "regress" else ""
        return f"{verb} '{spec.target}'{tail}, with a time-aware split, a baseline and validation{table}."
    if task == "forecast" and spec.measure and spec.time:
        unit = FREQ_WORDS[spec.freq][1 if (spec.horizon or 1) > 1 else 0]
        by = f", in total and by '{spec.group}'" if spec.group else ""
        return f"Forecast {_measure_label(spec)} per {FREQ_WORDS[spec.freq][0]} from '{spec.time}' for the next {spec.horizon} {unit}{by}{table}."
    if task == "why" and spec.measure and spec.time:
        against = "the same period last year" if spec.compare == "year" else "the previous period"
        rank = f"; rank '{spec.attention}' values that need attention" if spec.attention else ""
        return f"Explain the change in {_measure_label(spec)} in the latest complete {FREQ_WORDS[spec.freq][0]} vs {against}{rank}{table}."
    if task == "segment":
        unit = f"one unit per '{spec.unit}'" if spec.unit else "each row as a unit"
        features = f"features {', '.join(spec.features)}" if spec.features else "all suitable features"
        return f"Segment {unit}, using {features}; flag anomalies{table}."
    if task == "recommend" and spec.user and spec.item:
        when = "next month, day by day" if spec.period == "month" else f"the next {spec.days} days"
        return (f"Plan which '{spec.item}' each '{spec.user}' should contact {when}, ranked by the predicted chance "
                f"of a successful '{spec.outcome}'{table}.")
    if task == "ask":
        return f"Answer the question: \"{spec.question}\"."
    return TASKS.get(task, {}).get("label", "")
