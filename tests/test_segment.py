"""
Phase 3d: segmentation, anomaly detection, correlation.

Synthetic customers with three planted groups, five planted outliers, a
redundant feature and a column that is the exact sum of two others.
"""

import json
import time

import numpy as np
import pandas as pd
import plotly
import pytest
from sklearn.metrics import adjusted_rand_score

from core.schema_inference import infer_schema
from core.segment import SegmentSpec, build_segments, build_units
from service.session import DatasetBundle

OUTLIERS = ["CU900", "CU901", "CU902", "CU903", "CU904"]


def _customers(seed: int = 9) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    centres = {"A": (20, 5, 0.1), "B": (60, 30, 0.5), "C": (110, 8, 0.9)}
    rows = []
    for group, (spend, visits, share) in centres.items():
        for i in range(120):
            rows.append({
                "customer_code": f"CU{group}{i:03d}",
                "true_group": group,
                "spend": rng.normal(spend, 4),
                "visits": rng.normal(visits, 2),
                "online_share": float(np.clip(rng.normal(share, 0.05), 0, 1)),
                "channel": rng.choice(["Web", "Store"], p=[0.9, 0.1] if group == "C" else [0.2, 0.8]),
            })
    for code in OUTLIERS:
        rows.append({"customer_code": code, "true_group": "X", "spend": 900.0, "visits": 1.0,
                     "online_share": 0.5, "channel": "Store"})

    frame = pd.DataFrame(rows)
    frame["spend_doubled"] = frame["spend"] * 2 + rng.normal(0, 0.01, len(frame))
    frame["returns"] = rng.poisson(2, len(frame)).astype(float)
    frame["gross"] = frame["spend"] + frame["returns"]
    return frame


@pytest.fixture(scope="module")
def result():
    frame = _customers()
    features = ["spend", "visits", "online_share", "channel", "spend_doubled", "returns", "gross"]
    return frame, build_segments(frame, infer_schema(frame), SegmentSpec(features=features))


def test_per_entity_units_average_and_take_modes():
    panel = pd.DataFrame({
        "store_code": np.repeat(["SC01", "SC02", "SC03"], 4),
        "sales": [1, 2, 3, 4, 10, 10, 10, 10, 5, 5, 5, 5],
        "zone": ["N", "N", "S", "N", "S", "S", "S", "S", "E", "E", "E", "W"],
    })
    units = build_units(panel, infer_schema(panel), "store_code", ["sales", "zone"])

    assert list(units.frame["store_code"]) == ["SC01", "SC02", "SC03"]
    assert list(units.frame["sales"]) == [2.5, 10.0, 5.0]
    assert list(units.frame["zone"]) == ["N", "S", "E"]
    assert "Records" not in units.frame.columns       # every store has 4 rows


def test_kmeans_recovers_planted_groups(result):
    frame, outcome = result
    clustering = outcome.clustering

    assert clustering.k == 3
    keep = frame["true_group"] != "X"
    assert adjusted_rand_score(frame.loc[keep, "true_group"], clustering.labels[keep.to_numpy()]) > 0.9

    assert clustering.profiles["Units"].sum() == len(frame)
    assert clustering.profiles["Segment"].is_unique
    assert "Typical" not in set(clustering.profiles["Segment"])
    assert clustering.profiles["What sets it apart"].str.contains("Spend").any()


def test_planted_outliers_are_flagged_with_reasons():
    frame = _customers()
    outcome = build_segments(frame, infer_schema(frame), SegmentSpec(features=["spend", "visits", "online_share", "channel"], anomaly_share=0.02))

    flagged = set(outcome.anomalies["Row"])
    planted = set(frame.index[frame["customer_code"].isin(OUTLIERS)] + 1)
    assert len(flagged & planted) >= 4
    assert outcome.anomalies["Why unusual"].str.contains("Spend").any()


def test_correlation_report_finds_redundancy_and_exact_totals(result):
    _, outcome = result
    report = outcome.correlation

    pairs = {frozenset((a, b)) for a, b in zip(report.redundant["Feature A"], report.redundant["Feature B"])}
    assert frozenset(("spend", "spend_doubled")) in pairs

    vif = report.vif.set_index("Feature")
    assert "exact combination" in vif.loc["gross", "Note"] or vif.loc["gross", "VIF"] == "∞" or float(vif.loc["gross", "VIF"]) > 10
    assert any("redundant" in note for note in outcome.notes)


def test_bands_split_a_measure_into_thirds():
    frame = _customers()
    outcome = build_segments(frame, infer_schema(frame), SegmentSpec(method="bands", band_measure="spend", features=["spend", "visits"]))

    sizes = outcome.clustering.profiles.set_index("Segment")["Units"]
    assert set(sizes.index) == {"Low Spend", "Medium Spend", "High Spend"}
    assert sizes.max() - sizes.min() <= 1
    high = frame.loc[outcome.table["Segment"] == "High Spend", "spend"].min()
    low = frame.loc[outcome.table["Segment"] == "Low Spend", "spend"].max()
    assert high > low


def test_dashboard_tab_and_background_job():
    from ui import panels

    frame = _customers()
    bundle = DatasetBundle(id="s", name="customers.csv", content=b"", sheets=[], sheet=None,
                           df=frame, schema=infer_schema(frame))

    json.dumps(panels.segments_panel(bundle), cls=plotly.utils.PlotlyJSONEncoder)

    job_id = bundle.start_segment_job(None, SegmentSpec(features=["spend", "visits", "channel"]))
    deadline = time.time() + 120
    while bundle.model_job(job_id)["status"] == "running" and time.time() < deadline:
        time.sleep(0.3)

    job = bundle.model_job(job_id)
    assert job["status"] == "done", job["error"]
    json.dumps(panels.segments_results_view(job["result"]), cls=plotly.utils.PlotlyJSONEncoder)
