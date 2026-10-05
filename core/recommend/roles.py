"""
Who interacts with what, and what counts as success.

    user        the actor who makes the interaction      (MR, salesperson, customer)
    item        what the interaction is about             (doctor, store, product)
    time        when it happened                          (visit date)
    outcome     a category describing the result          (Outcome)
    success     outcome values that count as a success    (Interested, Order Placed)
    group       a shared attribute that limits who can be  (territory, region)
                assigned to what; may come from different
                columns on the user side and the item side

Everything is proposed from the data (roles, cardinalities, sheet links,
value names) and confirmed by the user in the dashboard.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd

from core.schema_inference import DatasetSchema
from core.target_analysis import native_grain

# Words in outcome values that usually mean a good result. Only used to
# pre-select success values; the user confirms them.
POSITIVE_WORDS = {
    "order", "ordered", "interested", "interest", "purchase", "purchased",
    "bought", "buy", "success", "successful", "yes", "won", "win",
    "converted", "conversion", "accepted", "approved", "sale", "sold",
    "positive", "booked", "signed", "renewed",
}

# Words that negate a positive word ("Not Interested", "Order cancelled").
NEGATIVE_WORDS = {
    "not", "no", "never", "non", "without", "lost", "lose", "cancel",
    "cancelled", "canceled", "declined", "rejected", "failed", "failure",
    "unsuccessful", "uninterested", "refused", "returned",
}

MAX_OUTCOME_VALUES = 20
# A planning group whose biggest value covers this share of rows is no group.
MAX_GROUP_SHARE = 0.8


@dataclass
class InteractionRoles:
    user: str
    item: str
    time: str
    outcome: str
    success_values: list[Any]
    user_group: str | None = None
    item_group: str | None = None
    item_attributes: list[str] = field(default_factory=list)
    user_sheet: str | None = None
    user_key: str | None = None
    item_sheet: str | None = None
    item_key: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {key: (list(value) if isinstance(value, list) else value) for key, value in self.__dict__.items()}


@dataclass
class RoleOptions:
    """Choices offered in the dashboard for each role."""

    users: list[str]
    items: list[str]
    times: list[str]
    outcomes: list[str]
    groups: list[tuple[str, str, str]]      # (label, user-side column, item-side column)


def role_options(
    frame: pd.DataFrame,
    schema: DatasetSchema,
    sources: dict[str, str],
) -> RoleOptions:

    base = [column for column in frame.columns if column not in sources]

    keys = [
        column for column in base
        if schema.role_of(column) == "identifier"
        and 1 < frame[column].nunique() < len(frame) * 0.5
    ]
    # Items usually have more distinct values than users (500 doctors, 50 MRs).
    keys = sorted(keys, key=lambda column: -frame[column].nunique())

    # The event date: filled on (almost) every row, recorded at irregular
    # moments rather than as a monthly snapshot, with many distinct values.
    times = sorted(
        [column for column in base if pd.api.types.is_datetime64_any_dtype(frame[column])],
        key=lambda column: (
            round(frame[column].notna().mean(), 2),
            native_grain(frame[column]) is None,
            frame[column].nunique(),
        ),
        reverse=True,
    )

    outcomes = [
        column for column in base
        if schema.role_of(column) in {"dimension", "binary"}
        and 2 <= frame[column].nunique() <= MAX_OUTCOME_VALUES
    ]
    outcomes = sorted(outcomes, key=lambda column: -len(suggest_success(frame[column])))

    return RoleOptions(
        users=keys[1:] + keys[:1],
        items=keys,
        times=times,
        outcomes=outcomes,
        groups=[],
    )


def group_options(
    frame: pd.DataFrame,
    sources: dict[str, str],
    user_sheet: str | None,
    item_sheet: str | None,
    user: str | None = None,
    item: str | None = None,
    exclude: tuple[str, ...] = (),
) -> list[tuple[str, str, str]]:
    """
    Attributes both sides share: the same column name coming from the
    user's sheet and the item's sheet ("MRs Territory" / "Doctors
    Territory"), or a column of the interaction table that is (nearly)
    constant for each user and for each item, such as a region.
    """

    def side(sheet: str | None) -> dict[str, str]:
        if not sheet:
            return {}
        return {
            _original(column, sheet): column
            for column, source in sources.items()
            if source == sheet
        }

    user_side, item_side = side(user_sheet), side(item_sheet)
    options = []

    for name in user_side:
        if name in item_side:
            user_column, item_column = user_side[name], item_side[name]
            if 1 < frame[item_column].nunique() <= 200:
                options.append((name, user_column, item_column))

    used = {o[1] for o in options} | {o[2] for o in options} | {user, item, *exclude}
    inline = []

    for column in frame.columns:
        if column in sources or column in used:
            continue
        values = frame[column]
        if values.dtype.kind in "biufcmM" or not 1 < values.nunique() <= 50:
            continue
        # A group splits items into real parts; a column where one value
        # covers most rows (e.g. a status that is almost always the same)
        # is constant per key only because it barely varies.
        if values.value_counts(normalize=True).iloc[0] >= MAX_GROUP_SHARE:
            continue
        purity = _purity(frame, column, user) + _purity(frame, column, item)
        if purity >= 1.6:
            inline.append((purity, column))

    options += [(column, column, column) for _, column in sorted(inline, reverse=True)]
    return options


def _purity(frame: pd.DataFrame, column: str, key: str | None) -> float:
    """Average share of each key's rows that carry its most common value."""

    if not key or key not in frame.columns:
        return 1.0
    counts = frame.groupby(key)[column].agg(lambda s: s.value_counts(normalize=True).iloc[0] if s.notna().any() else 0.0)
    return float(counts.mean()) if len(counts) else 0.0


def suggest_success(series: pd.Series) -> list[Any]:
    values = [value for value in series.dropna().unique()]
    chosen = []
    for value in values:
        words = set(re.findall(r"[a-z]+", str(value).lower()))
        if words & POSITIVE_WORDS and not words & NEGATIVE_WORDS:
            chosen.append(value)
    return chosen


def default_roles(
    frame: pd.DataFrame,
    schema: DatasetSchema,
    sources: dict[str, str],
    links: list,
) -> InteractionRoles:
    options = role_options(frame, schema, sources)

    if len(options.items) < 2 or not options.times or not options.outcomes:
        raise ValueError(
            "Recommendation needs an interaction table: two repeated key "
            "columns (who and what), a date column and an outcome column."
        )

    item, user = options.items[0], options.items[1]
    return complete_roles(
        frame, schema, sources, links,
        user=user, item=item, time=options.times[0], outcome=options.outcomes[0],
    )


def complete_roles(
    frame: pd.DataFrame,
    schema: DatasetSchema,
    sources: dict[str, str],
    links: list,
    user: str,
    item: str,
    time: str,
    outcome: str,
    success_values: list[Any] | None = None,
    group: str | None = "auto",
) -> InteractionRoles:
    """Fill in lookup sheets, group columns and item attributes for chosen roles."""

    if len({user, item, time, outcome}) < 4:
        raise ValueError("User, item, date and outcome must be four different columns.")

    user_link = _link(links, user)
    item_link = _link(links, item)
    user_sheet = user_link.lookup_sheet if user_link else None
    item_sheet = item_link.lookup_sheet if item_link else None

    groups = group_options(frame, sources, user_sheet, item_sheet, user, item, (time, outcome))
    chosen = None
    if group == "auto":
        chosen = groups[0] if groups else None
    elif group:
        chosen = next((option for option in groups if option[0] == group), None)

    if success_values is None:
        success_values = suggest_success(frame[outcome])
    if not success_values:
        raise ValueError(f"Choose which values of {outcome} count as a success.")

    return InteractionRoles(
        user=user,
        item=item,
        time=time,
        outcome=outcome,
        success_values=list(success_values),
        user_group=chosen[1] if chosen else None,
        item_group=chosen[2] if chosen else None,
        item_attributes=_item_attributes(frame, schema, sources, item, item_sheet, chosen),
        user_sheet=user_sheet,
        user_key=user_link.lookup_column if user_link else None,
        item_sheet=item_sheet,
        item_key=item_link.lookup_column if item_link else None,
    )


def _link(links: list, column: str):
    return next((link for link in links if link.column == column), None)


def _original(column: str, sheet: str) -> str:
    prefix = f"{sheet} "
    return column[len(prefix):] if column.startswith(prefix) else column


def _item_attributes(
    frame: pd.DataFrame,
    schema: DatasetSchema,
    sources: dict[str, str],
    item: str,
    item_sheet: str | None,
    group: tuple[str, str, str] | None,
) -> list[str]:
    """Stable descriptive columns of an item (specialty, potential, tier...)."""

    if item_sheet:
        candidates = [column for column, source in sources.items() if source == item_sheet]
    else:
        candidates = [column for column in frame.columns if column not in {item}]
        per_item = frame.groupby(item)[candidates].nunique(dropna=True).max()
        candidates = [column for column in candidates if per_item.get(column, 2) <= 1]

    attributes = []
    for column in candidates:
        role = schema.role_of(column)
        if role == "measure" or role == "binary":
            attributes.append(column)
        elif role == "dimension" and frame[column].nunique() <= 50:
            attributes.append(column)

    if group and group[2] not in attributes and group[2] in frame.columns:
        attributes.append(group[2])

    return attributes


def lookup_column_name(enriched: str, sheet: str | None) -> str:
    """Column name in the lookup sheet for an enriched column."""
    return _original(enriched, sheet) if sheet else enriched


Lookup = Callable[[str], "pd.DataFrame | None"]
