"""
Daily plan: which items each user should contact on each working day.

Before planning (optional, on by default) every item gets one owner: the
user in its group who has contacted it most, balanced so each user of a
group gets a similar share. Then, for every day of the plan:
    1. score every item as of that day (history features + model)
    2. keep items not contacted (or already planned) within the minimum gap
    3. give each user their best eligible items (their own items when
       owners are used, otherwise dealt out across the group in turn)
       until they reach the daily capacity

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
    period: str = "days"               # days | month (a whole calendar month)
    one_owner: bool = True             # each item stays with one user


def resolve_settings(settings: PlanSettings, history: pd.DataFrame, roles: InteractionRoles) -> PlanSettings:
    """Fill automatic values from the interaction history."""

    times = history[roles.time]
    last = times.max().normalize()
    days = max(1, int(settings.days))

    if settings.period == "month":
        # The calendar month of the chosen start, or the month after the
        # last recorded contact.
        if settings.start is not None:
            anchor = pd.Timestamp(settings.start).normalize()
        else:
            anchor = last + pd.offsets.MonthBegin(1)
        start = anchor.replace(day=1)
        month_end = start + pd.offsets.MonthEnd(0)
        calendar = pd.date_range(start, month_end, freq="D")
        days = int(sum(1 for day in calendar if not settings.working_days_only or day.weekday() < 5))
    elif settings.start is not None:
        start = pd.Timestamp(settings.start).normalize()
    else:
        start = last + pd.Timedelta(days=1)

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

    return PlanSettings(start, days, capacity, gap, settings.working_days_only, settings.period, settings.one_owner)


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

    users_by_group = users.groupby("_group")[roles.user].apply(lambda s: sorted(s.unique()))
    owners = assign_owners(history, universe, users, roles) if settings.one_owner else {}
    days = plan_days(settings)
    daily_limit = _level_load(universe, users_by_group, owners, settings, days, roles)

    planned_on: dict = {}
    rows = []

    for day in days:
        snap = snapshot(history, universe, roles, day)
        snap["Probability"] = score(snap)

        recent = snap["DaysSinceLastContact"].fillna(np.inf) < settings.min_gap_days
        planned_recently = snap[roles.item].map(
            lambda item: item in planned_on and (day - planned_on[item]).days < settings.min_gap_days
        )
        eligible = snap[~recent & ~planned_recently].sort_values("Probability", ascending=False, kind="stable")

        for group, group_users in users_by_group.items():
            pool = eligible[eligible["_group"] == group] if group != NO_GROUP else eligible

            if owners:
                owner = pool[roles.item].map(owners)
                for user in group_users:
                    picks = pool[owner == user].head(daily_limit[user])
                    for slot, (_, item_row) in enumerate(picks.iterrows(), start=1):
                        planned_on[item_row[roles.item]] = day
                        rows.append({"_row": item_row, "Day": day, roles.user: user, "Slot": slot})
                continue

            picks = pool.head(daily_limit[group])
            for position, (_, item_row) in enumerate(picks.iterrows()):
                user = group_users[position % len(group_users)]
                slot = position // len(group_users) + 1
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


def _level_load(universe, users_by_group, owners, settings, days, roles) -> dict:
    """
    Daily limit per user (or per group when items are shared) that spreads
    the contacts the period allows evenly over its days, instead of using
    the full capacity early and leaving later days empty. Each item can be
    contacted about once per minimum gap over the period.
    """

    if not days:
        return {}

    span = (days[-1] - days[0]).days + 1
    per_item = max(1, -(-span // settings.min_gap_days))
    limits = {}

    for group, group_users in users_by_group.items():
        if group == NO_GROUP:
            items = universe[roles.item]
        else:
            items = universe.loc[universe["_group"] == group, roles.item]

        if owners:
            for user in group_users:
                owned = int(sum(1 for item in items if owners.get(item) == user))
                limits[user] = min(settings.capacity, max(1, -(-owned * per_item // len(days))))
        else:
            expected = len(items) * per_item
            limits[group] = min(settings.capacity * len(group_users), max(1, -(-expected // len(days))))

    return limits


def assign_owners(
    history: pd.DataFrame,
    universe: pd.DataFrame,
    users: pd.DataFrame,
    roles: InteractionRoles,
) -> dict:
    """
    One owner per item: within the item's group, the user with the most
    past contacts with it, while keeping shares balanced (no user gets
    more than an even share of the group's items, rounded up).
    """

    owners: dict = {}

    for group, members in users.groupby("_group")[roles.user]:
        members = sorted(members.unique())
        if group == NO_GROUP:
            items = universe[roles.item].tolist()
        else:
            items = universe.loc[universe["_group"] == group, roles.item].tolist()
        if not items or not members:
            continue

        counts = (
            history[history[roles.user].isin(members) & history[roles.item].isin(items)]
            .groupby([roles.item, roles.user]).size()
        )
        known_items = set(counts.index.get_level_values(0))
        quota = -(-len(items) // len(members))
        load = {user: 0 for user in members}

        def past(item) -> pd.Series:
            return counts.loc[item] if item in known_items else pd.Series(dtype=int)

        for item in sorted(items, key=lambda value: -int(past(value).max()) if value in known_items else 0):
            known = past(item)
            ranked = sorted(members, key=lambda user: (-int(known.get(user, 0)), load[user], user))
            choice = next((user for user in ranked if load[user] < quota), min(members, key=lambda user: load[user]))
            owners[item] = choice
            load[choice] += 1

    return owners


def user_summary(plan: pd.DataFrame, roles: InteractionRoles) -> pd.DataFrame:
    """Per user: days with contacts, contacts, distinct items, average probability."""

    if plan.empty:
        return pd.DataFrame()

    columns = [roles.user] + (["Group"] if "Group" in plan.columns else [])
    summary = plan.groupby(columns).agg(
        Days=("Day", "nunique"),
        Contacts=(roles.item, "size"),
        Items=(roles.item, "nunique"),
        AvgProbability=("Probability", "mean"),
    ).reset_index()
    summary = summary.rename(columns={"AvgProbability": "Avg probability"})
    summary["Avg probability"] = summary["Avg probability"].round(3)
    return summary


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
