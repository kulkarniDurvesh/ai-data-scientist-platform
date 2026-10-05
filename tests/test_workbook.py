"""
Phase 1.5 slice 2: questions across the sheets of one workbook.

The synthetic workbook uses its own sheet and column names (shipments,
stores, couriers), so nothing here depends on one project's schema.
"""

import io

import numpy as np
import pandas as pd

from core.workbook import Workbook
from service.session import DatasetStore


def _workbook_bytes() -> bytes:
    rng = np.random.default_rng(3)

    stores = pd.DataFrame({
        "StoreCode": [f"ST{i:03d}" for i in range(1, 31)],
        "StoreName": [f"Store {i}" for i in range(1, 31)],
        "Format": rng.choice(["Mall", "Street", "Outlet"], 30),
        "Zone": rng.choice(["North", "South"], 30),
    })
    couriers = pd.DataFrame({
        "CourierKey": [f"CR{i:03d}" for i in range(1, 9)],
        "CourierName": [f"Courier {i}" for i in range(1, 9)],
        "Zone": ["North", "South"] * 4,
    })
    rows = 600
    shipments = pd.DataFrame({
        "ShipmentNo": np.arange(1, rows + 1),
        "StoreCode": rng.choice(stores["StoreCode"], rows),
        "CourierKey": rng.choice(couriers["CourierKey"], rows),
        "ShippedOn": pd.Timestamp("2024-01-01")
        + pd.to_timedelta(rng.integers(0, 365, rows), unit="D"),
        "Parcels": rng.integers(1, 20, rows),
        "Status": rng.choice(["Delivered", "Late", "Lost"], rows),
    })
    # A per-store summary with no courier column (like a training table).
    summary = (
        shipments.groupby("StoreCode")
        .agg(TotalParcels=("Parcels", "sum"), LateCount=("Status", lambda s: (s == "Late").sum()))
        .reset_index()
    )
    summary["Month"] = pd.Timestamp("2024-06-01")
    summary["ExtraA"], summary["ExtraB"], summary["ExtraC"] = 1.5, 2.5, 3.5

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer) as writer:
        summary.to_excel(writer, sheet_name="StoreSummary", index=False)
        shipments.to_excel(writer, sheet_name="Shipments", index=False)
        stores.to_excel(writer, sheet_name="Stores", index=False)
        couriers.to_excel(writer, sheet_name="Couriers", index=False)
    return buffer.getvalue()


def _frames():
    content = _workbook_bytes()
    shipments = pd.read_excel(io.BytesIO(content), sheet_name="Shipments")
    stores = pd.read_excel(io.BytesIO(content), sheet_name="Stores")
    return content, shipments, stores


def test_links_are_found_from_shared_key_columns():
    content = _workbook_bytes()
    workbook = Workbook(content, "book.xlsx", ["StoreSummary", "Shipments", "Stores", "Couriers"])

    pairs = {(link.sheet, link.column, link.lookup_sheet) for link in workbook.links()}

    assert ("Shipments", "StoreCode", "Stores") in pairs
    assert ("Shipments", "CourierKey", "Couriers") in pairs
    assert ("StoreSummary", "StoreCode", "Stores") in pairs


def test_clashing_lookup_columns_are_labelled_by_sheet():
    content = _workbook_bytes()
    workbook = Workbook(content, "book.xlsx", ["StoreSummary", "Shipments", "Stores", "Couriers"])

    enriched = workbook.enriched("Shipments")

    assert len(enriched.df) == 600
    assert {"Stores Zone", "Couriers Zone", "StoreName", "CourierName"} <= set(enriched.df.columns)
    assert "Zone" not in enriched.df.columns
    assert enriched.sources["StoreName"] == "Stores"


def test_question_is_routed_to_the_sheet_that_can_answer_it():
    content, shipments, stores = _frames()
    bundle = DatasetStore().load(content, "book.xlsx", "StoreSummary")

    entry = bundle.ask("list stores served by CR01 in May 2024")

    assert entry["error"] is None and entry["notice"] is None, entry
    mask = (
        (shipments["CourierKey"] == "CR001")
        & (shipments["ShippedOn"].dt.month == 5)
        & (shipments["ShippedOn"].dt.year == 2024)
    )
    assert len(entry["table"]) == shipments.loc[mask, "StoreCode"].nunique()
    assert "StoreName" in entry["table"].columns
    assert "'Shipments' sheet" in entry["answer"]


def test_lookup_columns_filter_and_disambiguate_values():
    content, shipments, stores = _frames()
    bundle = DatasetStore().load(content, "book.xlsx", "Shipments")

    entry = bundle.ask("how many shipments went to stores in North")

    north_stores = set(stores.loc[stores["Zone"] == "North", "StoreCode"])
    expected = shipments["StoreCode"].isin(north_stores).sum()
    assert entry["error"] is None, entry["error"]
    assert str(expected) in entry["answer"]
    assert "Stores Zone" in entry["answer"]


def test_chart_from_a_linked_sheet_renders_on_that_sheet():
    content, shipments, stores = _frames()
    bundle = DatasetStore().load(content, "book.xlsx", "StoreSummary")

    entry = bundle.ask("pie chart of shipments by status for CR002")

    assert entry["error"] is None, entry["error"]
    rendered = bundle.render(entry["chart_key"])
    expected = (
        shipments[shipments["CourierKey"] == "CR002"]
        .groupby("Status")["ShipmentNo"].nunique().to_dict()
    )
    assert {row["Status"]: row[rendered.spec.y_column] for row in rendered.data} == expected
