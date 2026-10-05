"""
Daily plan: which items each user should contact on each working day.

For every day of the plan:
    1. score every item as of that day (history features + model)
    2. keep items not contacted (or already planned) within the minimum gap
    3. within each group, deal the best items out to the group's users in
       turn until every user reaches the daily capacity

Rules always hold: capacity per user per day, minimum days between
contacts with the same item, and users only get items of their own group.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from .history import snapshot
from .roles import InteractionRoles

NO_GROUP = "(all)"


@dataclass
class PlanSettings:
    start: pd.Timestamp | None = None
    days: int = 5
    capacity: int | None = None        # contacts per user per day; None = from history
    min_gap_days: int | None = None    # None = from history
    working_days_only: bool = True


def resolve_settings(settings: PlanSettings, history: pd.DataFrame, roles: InteractionRoles) -> PlanSettings:
    """Fill automatic values from the interaction history."""

    times = history[roles.time]

    start = settings.start
    if start is None:
        start = (times.max() + pd.Timedelta(days=1)).normalize()
    if settings.working_days_only:
        while start.weekday() >= 5:
            start += pd.Timedelta(days=1)

    capacity = settings.capacity
    if not capacity:
        per_day = history.groupby([roles.user, times.dt.normalize()]).size()
        capacity = int(max(1, round(per_day.median())))

    gap = settings.min_gap_days
    if gap is None:
        gaps = history["DaysSinceLastContact"].dropna()
        gap = int(max(1, round(gaps.median()))) if len(gaps) else 7

    return PlanSettings(start, max(1, int(settings.days)), capacity, gap, settings.working_days_only)


def plan_days(settings: PlanSettings) -> list[pd.Timestamp]:
    days, day = [], settings.start
    while len(days) < settings.days:
        if not settings.working_days_only or day.weekday() < 5:
            days.append(day)
        day += pd.Timedelta(days=1)
    return days


def build_plan(
    history: pd.DataFrame,
    universe: pd.DataFrame,
    users: pd.DataFrame,
    roles: InteractionRoles,
    settings: PlanSettings,
    score: Callable[[pd.DataFrame], np.ndarray],
    explain: Callable[[pd.DataFrame, np.ndarray], list[str]],
) -> pd.DataFrame:

    group_of_item = universe.set_index(roles.item)["_group"]
    users_by_group = users.groupby("_group")[roles.user].apply(lambda s: sorted(s.unique()))

    planned_on: dict = {}
    rows = []

    for day in plan_days(settings):
        snap = snapshot(history, universe, roles, day)
        snap["Probability"] = score(snap)

        recent = snap["DaysSinceLastContact"].fillna(np.inf) < settings.min_gap_days
        planned_recently = snap[roles.item].map(
            lambda item: item in planned_on and (day - planned_on[item]).days < settings.min_gap_days
        )
        eligible = snap[~recent & ~planned_recently].sort_values("Probability", ascending=False, kind="stable")

        for group, group_users in users_by_group.items():
            pool = eligible[eligible["_group"] == group] if group != NO_GROUP else eligible
            pool = pool[~pool[roles.item].isin([k for k, d in planned_on.items() if d == day])]
            picks = pool.head(settings.capacity * len(group_users))

            for position, (_, item_row) in enumerate(picks.iterrows()):
                user = group_users[position % len(group_users)]
                slot = position // len(group_users) + 1
                if slot > settings.capacity:
                    break
                planned_on[item_row[roles.item]] = day
                rows.append({"_row": item_row, "Day": day, roles.user: user, "Slot": slot})

    if not rows:
        return pd.DataFrame()

    picked = pd.DataFrame([row["_row"] for row in rows]).reset_index(drop=True)
    reasons = explain(picked, picked["Probability"].to_numpy())

    plan = pd.DataFrame({
        "Day": [row["Day"] for row in rows],
        roles.user: [row[roles.user] for row in rows],
        "Slot": [row["Slot"] for row in rows],
    })
    if roles.user_group:
        plan["Group"] = picked["_group"].to_numpy()
    plan[roles.item] = picked[roles.item].to_numpy()

    for column in picked.columns:
        if column.startswith("_label:"):
            plan[column.split(":", 1)[1]] = picked[column].to_numpy()

    plan["Probability"] = picked["Probability"].round(4).to_numpy()
    plan["Days since last contact"] = picked["DaysSinceLastContact"].to_numpy()
    plan["Reasons"] = reasons

    return plan.sort_values(["Day", roles.user, "Slot"], kind="stable").reset_index(drop=True)


def plan_summary(plan: pd.DataFrame, users: pd.DataFrame, roles: InteractionRoles) -> pd.DataFrame:
    if plan.empty:
        return pd.DataFrame()

    key = "Group" if "Group" in plan.columns else None
    grouped = plan.groupby(key) if key else plan.groupby(lambda _: NO_GROUP)

    summary = grouped.agg(
        Users=(roles.user, "nunique"),
        Contacts=(roles.item, "size"),
        Items=(roles.item, "nunique"),
        AvgProbability=("Probability", "mean"),
    ).reset_index()
    summary = summary.rename(columns={"index": "Group", "AvgProbability": "Avg probability"})
    summary["Avg probability"] = summary["Avg probability"].round(3)
    return summary
