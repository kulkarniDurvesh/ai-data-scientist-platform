"""
Recommendation pipeline.

    interaction table + roles
        -> point-in-time history features per interaction
        -> success model (Phase 3a builder: time split, candidates, baseline)
        -> item and user universes (from lookup sheets when available)
        -> daily plan per user within their group
        -> backtest of prioritisation strategies on the held-out weeks
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from core.modeling import GoalSpec, ModelResult, build_model
from core.modeling.explain import ReasonBuilder
from core.modeling.features import prepare_frame
from core.schema_inference import infer_schema, name_tokens

from .backtest import backtest
from .history import HISTORY_FEATURES, SUCCESS, training_frame
from .planner import NO_GROUP, PlanSettings, build_plan, plan_summary, resolve_settings, user_summary
from .roles import InteractionRoles, lookup_column_name

LABEL_TOKENS = {"name", "title", "label"}

Progress = Callable[[str], None]
Lookup = Callable[[str], "pd.DataFrame | None"]


@dataclass
class RecommendResult:
    roles: InteractionRoles
    settings: PlanSettings
    model: ModelResult
    plan: pd.DataFrame
    summary_table: pd.DataFrame
    user_table: pd.DataFrame
    backtest: pd.DataFrame
    n_items: int
    n_users: int
    summary: str = ""
    warnings: list[str] = field(default_factory=list)
    seconds: float = 0.0


def build_recommender(
    frame: pd.DataFrame,
    roles: InteractionRoles,
    settings: PlanSettings | None = None,
    lookup: Lookup | None = None,
    progress: Progress | None = None,
) -> RecommendResult:

    started = time.time()
    say = progress or (lambda message: None)
    lookup = lookup or (lambda sheet: None)
    notes: list[str] = []

    say("Building point-in-time history features")
    history = training_frame(frame, roles)

    if history[SUCCESS].nunique() < 2:
        raise ValueError("Every contact has the same outcome: choose success values that split them.")

    features = HISTORY_FEATURES + [c for c in roles.item_attributes if c in history.columns]
    columns = list(dict.fromkeys([roles.item, roles.time, SUCCESS] + features))
    train = history[columns]
    schema = infer_schema(train)

    spec = GoalSpec(
        goal_type="rank",
        target=SUCCESS,
        features=features,
        time_column=roles.time,
        entity=roles.item,
    )

    model = build_model(train, schema, spec, progress=say)

    say("Preparing items and users")
    universe = _item_universe(history, roles, lookup)
    users = _user_universe(history, roles, lookup, set(universe["_group"]))

    if users.empty:
        raise ValueError("No users share a group with any item; try planning without a group.")

    resolved = resolve_settings(settings or PlanSettings(), history, roles)

    labelled = history.index
    reference = prepare_frame(history.loc[labelled], model.numeric, model.categorical)
    reasons = ReasonBuilder(
        reference, model.predict(history.loc[labelled]), model.importance, model.numeric, model.categorical,
    )

    say("Planning visits")
    plan = build_plan(
        history, universe, users, roles, resolved,
        score=model.predict,
        explain=lambda rows, scores: reasons.explain(
            prepare_frame(rows, model.numeric, model.categorical), scores,
        ),
    )

    say("Backtesting strategies")
    table = backtest(history, roles.time, model.test_scores)

    result = RecommendResult(
        roles=roles,
        settings=resolved,
        model=model,
        plan=plan,
        summary_table=plan_summary(plan, users, roles),
        user_table=user_summary(plan, roles),
        backtest=table,
        n_items=len(universe),
        n_users=len(users),
        warnings=notes + model.warnings,
        seconds=round(time.time() - started, 1),
    )
    result.summary = _summarize(result)

    slots = resolved.capacity * len(users) * resolved.days
    if plan.empty:
        result.warnings.append("No contacts could be planned: lower the minimum gap or check the groups.")
    elif len(plan) < slots:
        per_user = len(universe) / max(len(users), 1)
        span = (plan["Day"].max() - resolved.start).days + 1
        per_item = max(1, -(-span // resolved.min_gap_days))
        result.warnings.append(
            f"{len(plan):,} contacts planned, while capacity allows {slots:,}. Each user "
            f"has about {per_user:.0f} items, and an item can be contacted about "
            f"{per_item} time(s) in this period with a {resolved.min_gap_days}-day gap, "
            f"so the number of items per user — not capacity — limits the plan. "
            f"Contacts are spread evenly over the days; lower the gap to contact items more often."
        )

    say("Done")
    return result


def _item_universe(history: pd.DataFrame, roles: InteractionRoles, lookup: Lookup) -> pd.DataFrame:
    """Every item that can be planned, with its attributes, group and label."""

    attributes = [c for c in roles.item_attributes if c in history.columns]
    from_history = history.groupby(roles.item)[attributes].first().reset_index()

    sheet = lookup(roles.item_sheet) if roles.item_sheet else None
    universe = from_history

    if sheet is not None and roles.item_key in sheet.columns:
        listed = pd.DataFrame({roles.item: sheet[roles.item_key].to_numpy()})
        for column in attributes:
            original = lookup_column_name(column, roles.item_sheet)
            if original in sheet.columns:
                listed[column] = sheet[original].to_numpy()
        missing = from_history[~from_history[roles.item].isin(listed[roles.item])]
        universe = pd.concat([listed, missing], ignore_index=True).drop_duplicates(roles.item)

        label = _label_column(sheet, roles.item_key)
        if label:
            universe[f"_label:{label}"] = universe[roles.item].map(
                dict(zip(sheet[roles.item_key], sheet[label]))
            )

    if roles.item_group:
        if roles.item_group in universe.columns and universe[roles.item_group].notna().any():
            universe["_group"] = universe[roles.item_group]
        else:
            modes = history.groupby(roles.item)[roles.item_group].agg(_mode)
            universe["_group"] = universe[roles.item].map(modes)
        universe = universe.dropna(subset=["_group"])
    else:
        universe["_group"] = NO_GROUP

    return universe.reset_index(drop=True)


def _user_universe(
    history: pd.DataFrame,
    roles: InteractionRoles,
    lookup: Lookup,
    groups: set,
) -> pd.DataFrame:

    sheet = lookup(roles.user_sheet) if roles.user_sheet else None

    if roles.user_group:
        original = lookup_column_name(roles.user_group, roles.user_sheet)
        if sheet is not None and roles.user_key in sheet.columns and original in sheet.columns:
            users = pd.DataFrame({roles.user: sheet[roles.user_key], "_group": sheet[original]})
        else:
            users = history.groupby(roles.user)[roles.user_group].agg(_mode).rename("_group").reset_index()
        users = users[users["_group"].isin(groups)]
    else:
        names = sheet[roles.user_key] if sheet is not None and roles.user_key in sheet.columns else history[roles.user]
        users = pd.DataFrame({roles.user: pd.unique(names.dropna()), "_group": NO_GROUP})

    return users.dropna().drop_duplicates(roles.user).reset_index(drop=True)


def _label_column(sheet: pd.DataFrame, key: str) -> str | None:
    """A readable name for items, e.g. DoctorName (one distinct value per item)."""

    for column in sheet.columns:
        if column == key or sheet[column].dtype.kind in "biufcmM":
            continue
        if set(name_tokens(column)) & LABEL_TOKENS and sheet[column].nunique() == sheet[key].nunique():
            return column
    return None


def _mode(series: pd.Series):
    values = series.dropna()
    return values.mode().iloc[0] if len(values) else np.nan


def _summarize(result: RecommendResult) -> str:
    table = result.backtest.set_index("Strategy")
    model, random = table.loc["Model"], table.loc["Random"]
    best_rule = table.drop(index=["Model", "Random"])["Success rate"].idxmax()
    share = 20
    settings = result.settings

    return (
        f"Backtest on the held-out weeks: contacts in the top {share}% by the model "
        f"succeeded {model['Success rate']:.1%} of the time, versus "
        f"{random['Success rate']:.1%} without prioritisation ({model['Lift']:.1f}× lift) and "
        f"{table.loc[best_rule, 'Success rate']:.1%} for the best simple rule "
        f"({best_rule.lower()}). Plan: {len(result.plan):,} contacts for "
        f"{result.n_users} users over {settings.days} working day(s) from "
        f"{settings.start:%Y-%m-%d}"
        + (f" (all of {settings.start:%B %Y})" if settings.period == "month" else "")
        + f", up to {settings.capacity} per user per day, at least "
        f"{settings.min_gap_days} days between contacts with the same item"
        + (", each item kept with one user of its group." if settings.one_owner else ".")
    )
