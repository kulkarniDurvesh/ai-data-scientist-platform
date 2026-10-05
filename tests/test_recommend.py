"""
Phase 3b: recommendation (next-best-contact model, daily plans, backtest).

A synthetic sales workbook (reps, accounts, contacts) with its own names,
so nothing here depends on the pharma schema.
"""

import io
import json
import time

import numpy as np
import pandas as pd
import plotly
import pytest

from core.recommend import PlanSettings, build_recommender, default_roles
from core.recommend.history import training_frame
from core.recommend.roles import InteractionRoles
from core.workbook import Workbook
from ui.state import DatasetStore

SHEETS = ["Contacts", "Reps", "Accounts"]


def _workbook() -> bytes:
    rng = np.random.default_rng(5)
    regions = ["East", "West", "Central"]

    reps = pd.DataFrame({
        "RepCode": [f"RP{i:02d}" for i in range(1, 10)],
        "Region": np.repeat(regions, 3),
    })
    accounts = pd.DataFrame({
        "AccountKey": [f"AC{i:03d}" for i in range(1, 151)],
        "AccountName": [f"Account {i}" for i in range(1, 151)],
        "Region": rng.choice(regions, 150),
        "Tier": rng.choice(["Gold", "Silver", "Bronze"], 150, p=[0.2, 0.4, 0.4]),
    })
    propensity = rng.beta(2, 5, 150) + np.where(accounts["Tier"] == "Gold", 0.35, 0.0)

    rows = []
    for _ in range(6000):
        account = rng.integers(0, 150)
        region = accounts.loc[account, "Region"]
        rep = rng.choice(reps.loc[reps["Region"] == region, "RepCode"])
        day = pd.Timestamp("2024-01-01") + pd.Timedelta(days=int(rng.integers(0, 420)))
        won = rng.random() < min(0.95, propensity[account])
        outcome = rng.choice(["Deal Won", "Interested"]) if won else rng.choice(["No Interest", "Cancelled"])
        rows.append((rep, accounts.loc[account, "AccountKey"], day, outcome, day + pd.Timedelta(days=30)))

    contacts = pd.DataFrame(rows, columns=["RepCode", "AccountKey", "ContactedOn", "Result", "FollowUpOn"])
    contacts.loc[contacts.sample(frac=0.6, random_state=1).index, "FollowUpOn"] = pd.NaT

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer) as writer:
        contacts.to_excel(writer, sheet_name="Contacts", index=False)
        reps.to_excel(writer, sheet_name="Reps", index=False)
        accounts.to_excel(writer, sheet_name="Accounts", index=False)
    return buffer.getvalue()


@pytest.fixture(scope="module")
def setup():
    content = _workbook()
    workbook = Workbook(content, "sales.xlsx", SHEETS)
    enriched = workbook.enriched("Contacts")
    from core.schema_inference import infer_schema

    schema = infer_schema(enriched.df)
    links = [link for link in workbook.links() if link.sheet == "Contacts"]
    roles = default_roles(enriched.df, schema, enriched.sources, links)
    return content, workbook, enriched.df, roles


@pytest.fixture(scope="module")
def result(setup):
    _, workbook, frame, roles = setup
    return build_recommender(frame, roles, PlanSettings(days=5, capacity=4), lookup=workbook.frame)


def test_roles_are_detected(setup):
    _, _, _, roles = setup

    assert (roles.user, roles.item, roles.time, roles.outcome) == ("RepCode", "AccountKey", "ContactedOn", "Result")
    assert set(roles.success_values) == {"Deal Won", "Interested"}      # "No Interest" is not a success
    assert (roles.user_group, roles.item_group) == ("Reps Region", "Accounts Region")
    assert "Tier" in roles.item_attributes


def test_history_features_use_only_the_past():
    frame = pd.DataFrame({
        "u": ["a"] * 4,
        "i": ["x", "x", "x", "y"],
        "t": pd.to_datetime(["2024-01-01", "2024-01-11", "2024-06-01", "2024-01-05"]),
        "o": ["win", "lose", "win", "win"],
    })
    roles = InteractionRoles(user="u", item="i", time="t", outcome="o", success_values=["win"])
    data = training_frame(frame, roles).set_index("t")

    x3 = data.loc[pd.Timestamp("2024-06-01")]
    assert x3["PriorContacts"] == 2 and x3["PriorSuccesses"] == 1
    assert x3["PriorSuccessRate"] == 0.5
    assert x3["DaysSinceLastContact"] == (pd.Timestamp("2024-06-01") - pd.Timestamp("2024-01-11")).days
    assert x3["DaysSinceLastSuccess"] == (pd.Timestamp("2024-06-01") - pd.Timestamp("2024-01-01")).days
    assert x3["LastContactSucceeded"] == 0
    assert x3["Contacts90d"] == 0

    x2 = data.loc[pd.Timestamp("2024-01-11")]
    assert x2["Contacts90d"] == 1 and x2["LastContactSucceeded"] == 1

    first = data.loc[pd.Timestamp("2024-01-01")]
    assert first["PriorContacts"] == 0 and pd.isna(first["PriorSuccessRate"])


def test_plan_respects_every_rule(setup, result):
    _, workbook, frame, roles = setup
    plan, settings = result.plan, result.settings

    assert not plan.empty
    assert plan.groupby(["Day", roles.user]).size().max() <= settings.capacity
    assert all(pd.Timestamp(day).weekday() < 5 for day in plan["Day"])

    reps = workbook.frame("Reps").set_index("RepCode")["Region"]
    accounts = workbook.frame("Accounts").set_index("AccountKey")["Region"]
    assert (plan[roles.user].map(reps) == plan[roles.item].map(accounts)).all()

    # No account twice within the gap, and none contacted within the gap before.
    assert plan[roles.item].is_unique or settings.days > settings.min_gap_days
    last_contact = frame.groupby(roles.item)[roles.time].max()
    gaps = (pd.to_datetime(plan["Day"]) - plan[roles.item].map(last_contact)).dt.days
    assert (gaps >= settings.min_gap_days).all()

    assert "AccountName" in plan.columns and plan["Reasons"].str.len().gt(0).any()


def test_backtest_beats_random(result):
    table = result.backtest.set_index("Strategy")

    assert table.loc["Model", "Success rate"] > table.loc["Random", "Success rate"] * 1.3
    assert abs(table.loc["Random", "Lift"] - 1) < 0.1
    assert "Most overdue first" in table.index and "Highest past success rate" in table.index


def test_single_table_without_lookup_sheets():
    content = _workbook()
    contacts = pd.read_excel(io.BytesIO(content), sheet_name="Contacts")
    accounts = pd.read_excel(io.BytesIO(content), sheet_name="Accounts")
    flat = contacts.merge(accounts[["AccountKey", "Region", "Tier"]], on="AccountKey")

    from core.schema_inference import infer_schema

    roles = default_roles(flat, infer_schema(flat), {}, [])
    assert roles.user_group == roles.item_group == "Region"

    result = build_recommender(flat, roles, PlanSettings(days=3, capacity=3))
    assert not result.plan.empty
    assert result.plan.groupby(["Day", roles.user]).size().max() <= 3


def test_dashboard_tab_and_background_job():
    from ui import panels

    bundle = DatasetStore().load(_workbook(), "sales.xlsx", "Accounts")
    assert bundle.interaction_sheets() == ["Contacts"]

    json.dumps(panels.recommend_panel(bundle), cls=plotly.utils.PlotlyJSONEncoder)

    roles = bundle.interaction_options("Contacts")[1]
    job_id = bundle.start_recommend_job("Contacts", roles, PlanSettings(days=2, capacity=2))

    deadline = time.time() + 120
    while bundle.model_job(job_id)["status"] == "running" and time.time() < deadline:
        time.sleep(0.5)

    job = bundle.model_job(job_id)
    assert job["status"] == "done", job["error"]
    assert bundle.latest_recommendation is job["result"]

    json.dumps(panels.recommend_results_view(job["result"]), cls=plotly.utils.PlotlyJSONEncoder)


def test_month_plan_keeps_each_account_with_one_rep_and_spreads_visits(setup):
    _, workbook, frame, roles = setup
    result = build_recommender(
        frame, roles,
        PlanSettings(period="month", start=pd.Timestamp("2025-03-10"), capacity=5, min_gap_days=7),
        lookup=workbook.frame,
    )
    plan, settings = result.plan, result.settings

    # The whole calendar month of the start date, working days only.
    assert settings.start == pd.Timestamp("2025-03-03")
    assert settings.days == 21
    assert pd.to_datetime(plan["Day"]).dt.month.eq(3).all()

    # One rep per account, same region, visits at least the gap apart.
    assert (plan.groupby(roles.item)[roles.user].nunique() == 1).all()
    reps = workbook.frame("Reps").set_index("RepCode")["Region"]
    accounts = workbook.frame("Accounts").set_index("AccountKey")["Region"]
    assert (plan[roles.user].map(reps) == plan[roles.item].map(accounts)).all()
    spacing = plan.sort_values("Day").groupby(roles.item)["Day"].diff().dropna().dt.days
    assert (spacing >= 7).all()

    # Load is levelled: no rep gets the full capacity early and nothing later.
    per_day = plan.groupby([roles.user, "Day"]).size()
    assert per_day.max() <= settings.capacity
    active_days = plan.groupby(roles.user)["Day"].nunique()
    assert active_days.min() >= settings.days // 2

    assert set(result.user_table[roles.user]) == set(plan[roles.user])


def test_shared_items_can_go_to_any_rep_of_the_region(setup):
    _, workbook, frame, roles = setup
    result = build_recommender(
        frame, roles, PlanSettings(days=10, capacity=4, min_gap_days=3, one_owner=False),
        lookup=workbook.frame,
    )
    assert (result.plan.groupby(roles.item)[roles.user].nunique() > 1).any()

