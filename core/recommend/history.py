"""
Point-in-time history features.

For every interaction, features are computed only from earlier
interactions with the same item, so a model trained on them never sees
the future. The same features are computed "as of" any date for every
item, which is how a plan is scored.

    DaysSinceLastContact   days since the item was last contacted
    DaysSinceLastSuccess   days since the last successful contact
    PriorContacts          contacts before this one
    PriorSuccesses         successful contacts before this one
    PriorSuccessRate       PriorSuccesses / PriorContacts
    Contacts90d            contacts in the previous 90 days
    LastContactSucceeded   1 if the previous contact succeeded
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .roles import InteractionRoles

WINDOW_DAYS = 90
SUCCESS = "Success"

HISTORY_FEATURES = [
    "DaysSinceLastContact",
    "DaysSinceLastSuccess",
    "PriorContacts",
    "PriorSuccesses",
    "PriorSuccessRate",
    "Contacts90d",
    "LastContactSucceeded",
]


def training_frame(frame: pd.DataFrame, roles: InteractionRoles) -> pd.DataFrame:
    """One row per interaction with its outcome and history features."""

    columns = list(dict.fromkeys(
        [roles.user, roles.item, roles.time, roles.outcome]
        + roles.item_attributes
        + [c for c in (roles.user_group, roles.item_group) if c]
    ))

    data = frame[columns].dropna(subset=[roles.item, roles.time]).copy()
    data[SUCCESS] = data[roles.outcome].isin(roles.success_values).astype(int)
    data = data.sort_values([roles.item, roles.time], kind="stable")

    by_item = data.groupby(roles.item, sort=False)
    times = data[roles.time]

    data["DaysSinceLastContact"] = (times - by_item[roles.time].shift(1)).dt.days
    data["PriorContacts"] = by_item.cumcount()
    data["PriorSuccesses"] = by_item[SUCCESS].cumsum() - data[SUCCESS]
    data["PriorSuccessRate"] = (data["PriorSuccesses"] / data["PriorContacts"]).where(data["PriorContacts"] > 0)
    data["LastContactSucceeded"] = by_item[SUCCESS].shift(1)

    success_times = times.where(data[SUCCESS] == 1)
    last_success = success_times.groupby(data[roles.item]).transform(lambda s: s.shift(1).ffill())
    data["DaysSinceLastSuccess"] = (times - last_success).dt.days

    data["Contacts90d"] = by_item[roles.time].transform(_window_counts)

    return data


def _window_counts(times: pd.Series) -> pd.Series:
    values = times.to_numpy(dtype="datetime64[ns]")
    start = np.searchsorted(values, values - np.timedelta64(WINDOW_DAYS, "D"), side="left")
    return pd.Series(np.arange(len(values)) - start, index=times.index)


def snapshot(
    history: pd.DataFrame,
    universe: pd.DataFrame,
    roles: InteractionRoles,
    as_of: pd.Timestamp,
) -> pd.DataFrame:
    """History features for every item in `universe` as of a date."""

    past = history[history[roles.time] < as_of]
    grouped = past.sort_values(roles.time, kind="stable").groupby(roles.item)

    stats = pd.DataFrame({
        "PriorContacts": grouped.size(),
        "PriorSuccesses": grouped[SUCCESS].sum(),
        "_last": grouped[roles.time].max(),
        "LastContactSucceeded": grouped[SUCCESS].last(),
    })
    stats["_last_success"] = past[past[SUCCESS] == 1].groupby(roles.item)[roles.time].max()
    recent = past[past[roles.time] >= as_of - pd.Timedelta(days=WINDOW_DAYS)]
    stats["Contacts90d"] = recent.groupby(roles.item).size()

    out = universe.merge(stats, left_on=roles.item, right_index=True, how="left")
    out["PriorContacts"] = out["PriorContacts"].fillna(0)
    out["PriorSuccesses"] = out["PriorSuccesses"].fillna(0)
    out["Contacts90d"] = out["Contacts90d"].fillna(0)
    out["PriorSuccessRate"] = (out["PriorSuccesses"] / out["PriorContacts"]).where(out["PriorContacts"] > 0)
    out["DaysSinceLastContact"] = (as_of - out["_last"]).dt.days
    out["DaysSinceLastSuccess"] = (as_of - out["_last_success"]).dt.days

    return out.drop(columns=["_last", "_last_success"])
