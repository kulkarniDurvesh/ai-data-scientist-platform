"""
Phase 4b: KPI definitions layer.

A synthetic sales dataset (contacts, accounts) and a domain file written
in the test, so nothing depends on the pharma workbook. The pharma domain
file is only checked for being valid configuration.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly
import pytest
import yaml

from core.kpi import ConfigError, evaluate_domain, load_domain, parse_domain, starter_yaml
from core.schema_inference import infer_schema
from ui.state import DatasetBundle

DOMAIN = """
name: Sales test
roles:
  rep: RepCode
  account: AccountKey
  region:
    Contacts: Region
    Accounts: Region
  date:
    Contacts: ContactedOn
  result:
    Contacts: Result
kpis:
  - id: contacts
    table: Contacts
    value: {count: rows}
  - id: win_rate
    label: Win rate
    table: Contacts
    numerator: {count: rows, where: {result: Won}}
    denominator: {count: rows, where: {result: [Won, Lost]}}
    direction: up
  - id: cancellations
    label: Cancellation rate
    table: Contacts
    numerator: {count: rows, where: {result: Cancelled}}
    denominator: {count: rows}
    direction: down
  - id: accounts_reached
    table: Contacts
    value: {distinct: account}
  - id: coverage
    label: Coverage
    table: Contacts
    numerator: {distinct: account}
    denominator: {count: rows, table: Accounts, where: {Active: true}}
  - id: big_deals
    table: Contacts
    value: {count: rows, where: {Amount: {gt: 500}}}
  - id: broken
    table: Contacts
    value: {sum: NoSuchColumn}
"""


def _tables(seed: int = 3, end_mid_month: bool = False):
    rng = np.random.default_rng(seed)
    accounts = pd.DataFrame({
        "AccountKey": [f"AK{i:03d}" for i in range(60)],
        "Region": rng.choice(["East", "West"], 60),
        "Active": rng.random(60) > 0.1,
    })
    rows = []
    for month in pd.date_range("2024-01-01", periods=6, freq="MS"):
        for _ in range(200):
            account = accounts.sample(1, random_state=int(rng.integers(0, 1_000_000))).iloc[0]
            rows.append({
                "RepCode": f"RP{rng.integers(1, 6)}",
                "AccountKey": account["AccountKey"],
                "Region": account["Region"],
                "ContactedOn": month + pd.Timedelta(days=int(rng.integers(0, 28))),
                "Result": rng.choice(["Won", "Lost", "Cancelled"], p=[0.3 + 0.02 * month.month, 0.6 - 0.02 * month.month, 0.1]),
                "Amount": float(rng.integers(50, 900)),
            })
    contacts = pd.DataFrame(rows)
    if end_mid_month:
        late = contacts.copy()
        late["ContactedOn"] = pd.Timestamp("2024-07-01") + pd.to_timedelta(rng.integers(0, 5, len(late)), unit="D")
        contacts = pd.concat([contacts, late.head(40)], ignore_index=True)
    frames = {"Contacts": contacts, "Accounts": accounts}
    return frames, (lambda name: frames.get(name))


@pytest.fixture(scope="module")
def config():
    return parse_domain(yaml.safe_load(DOMAIN))


def test_invalid_files_are_explained():
    with pytest.raises(ConfigError, match="needs an 'id'"):
        parse_domain({"kpis": [{"value": {"count": "rows"}}]})
    with pytest.raises(ConfigError, match="exactly one"):
        parse_domain({"kpis": [{"id": "x", "value": {"median": "Amount"}}]})
    with pytest.raises(ConfigError, match="direction"):
        parse_domain({"kpis": [{"id": "x", "value": {"count": "rows"}, "direction": "sideways"}]})
    with pytest.raises(ConfigError, match="no KPIs"):
        parse_domain({"name": "empty", "roles": {}})


def test_kpis_match_hand_calculations(config):
    frames, tables = _tables()
    report = evaluate_domain(config, tables, "M")
    contacts = frames["Contacts"]
    june = contacts[contacts["ContactedOn"].dt.month == 6]
    may = contacts[contacts["ContactedOn"].dt.month == 5]

    assert (report.current_period, report.previous_period) == (pd.Timestamp("2024-06-01"), pd.Timestamp("2024-05-01"))
    rows = report.summary.set_index("KPI")

    win = (june["Result"] == "Won").sum() / june["Result"].isin(["Won", "Lost"]).sum()
    assert rows.loc["Win rate", "Current"] == pytest.approx(win)
    assert rows.loc["Contacts", "Current"] == len(june)
    assert rows.loc["Big Deals", "Current"] == (june["Amount"] > 500).sum()

    active = frames["Accounts"]["Active"].sum()
    assert rows.loc["Coverage", "Current"] == pytest.approx(june["AccountKey"].nunique() / active)

    # Direction decides better / worse.
    previous_win = (may["Result"] == "Won").sum() / may["Result"].isin(["Won", "Lost"]).sum()
    assert rows.loc["Win rate", "Status"] == ("better" if win > previous_win else "worse")
    june_cancel = (june["Result"] == "Cancelled").mean()
    may_cancel = (may["Result"] == "Cancelled").mean()
    assert rows.loc["Cancellation rate", "Status"] == ("worse" if june_cancel > may_cancel else "better")

    assert list(report.unavailable["KPI"]) == ["Broken"]
    assert "NoSuchColumn" in report.unavailable["Reason"].iloc[0]


def test_breakdown_by_role_including_cross_table_ratios(config):
    frames, tables = _tables()
    report = evaluate_domain(config, tables, "M", group_role="region")
    contacts, accounts = frames["Contacts"], frames["Accounts"]
    june = contacts[contacts["ContactedOn"].dt.month == 6]

    table = report.by_group.set_index("Region")
    for region in ("East", "West"):
        part = june[june["Region"] == region]
        assert table.loc[region, "Win rate"] == pytest.approx((part["Result"] == "Won").sum() / part["Result"].isin(["Won", "Lost"]).sum())
        active = accounts[(accounts["Region"] == region) & accounts["Active"]]
        assert table.loc[region, "Coverage"] == pytest.approx(part["AccountKey"].nunique() / len(active))

    by_rep = evaluate_domain(config, tables, "M", group_role="rep")
    reasons = by_rep.unavailable.set_index("KPI")["Reason"]
    assert "Coverage" in reasons.index and "Accounts" in reasons["Coverage"]


def test_partial_last_month_is_left_out(config):
    _, tables = _tables(end_mid_month=True)
    report = evaluate_domain(config, tables, "M")
    assert report.current_period == pd.Timestamp("2024-06-01")


def test_starter_file_is_valid_and_computes_on_its_dataset(tmp_path):
    frames, _ = _tables()
    contacts = frames["Contacts"]
    text = starter_yaml(contacts, infer_schema(contacts), {}, None, "contacts.csv")

    path = tmp_path / "starter.yaml"
    path.write_text(text, encoding="utf-8")
    starter = load_domain(path)

    assert "rep" in starter.roles and starter.roles["date"] == "ContactedOn"
    report = evaluate_domain(starter, lambda name: contacts if name is None else None, "M")
    assert len(report.summary) >= 3 and report.unavailable.empty
    assert any(label.startswith("Share Result") for label in report.summary["KPI"])


def test_pharma_domain_file_is_valid_configuration():
    config = load_domain(Path(__file__).resolve().parents[1] / "domains" / "pharma_sfa.yaml")
    assert len(config.kpis) >= 10 and "territory" in config.group_roles()


def test_kpi_panel_renders(tmp_path, config):
    from ui import panels

    frames, _ = _tables()
    contacts = frames["Contacts"]
    bundle = DatasetBundle(id="k", name="contacts.csv", content=b"", sheets=[], sheet=None,
                           df=contacts, schema=infer_schema(contacts))

    json.dumps(panels.kpi_panel(bundle), cls=plotly.utils.PlotlyJSONEncoder)

    path = tmp_path / "flat.yaml"
    flat = DOMAIN.replace("table: Contacts", "table: null").replace("    Contacts: ", "    '*': ")
    path.write_text(flat, encoding="utf-8")
    view = panels.kpi_results_view(bundle, str(path), "M", "region")
    json.dumps(view, cls=plotly.utils.PlotlyJSONEncoder)
