"""
Phase 2: target-aware EDA.

A synthetic panel (stores observed monthly) with planted signal,
leakage and a segment effect, using names unrelated to any project.
"""

import json

import numpy as np
import pandas as pd
import plotly

from core.schema_inference import infer_schema
from core.target_analysis import analyze_target, detect_targets, report_markdown
from service.session import DatasetBundle


def _panel(rows_per_store: int = 18, stores: int = 120, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    store = np.repeat([f"SC{i:03d}" for i in range(stores)], rows_per_store)
    period = np.tile(pd.date_range("2023-01-01", periods=rows_per_store, freq="MS"), stores)
    region = np.repeat(rng.choice(["Coast", "Hills", "Plains"], stores), rows_per_store)

    complaints = rng.poisson(2, len(store))
    footfall = rng.normal(500, 80, len(store)).round()

    # Churn rises with complaints and is higher on the coast.
    logit = -4.3 + 0.45 * complaints + np.where(region == "Coast", 0.9, 0.0)
    will_churn = (rng.random(len(store)) < 1 / (1 + np.exp(-logit))).astype(int)

    return pd.DataFrame({
        "store_code": store,
        "period": period,
        "region": region,
        "is_premium": rng.choice([0, 1], len(store), p=[0.7, 0.3]),
        "complaints": complaints,
        "footfall": footfall,
        # Planted leaks: a near-copy of the outcome and a future-named column.
        "refund_after_close": will_churn * rng.uniform(50, 100, len(store)),
        "closed_on": period + pd.to_timedelta(45, unit="D"),
        "will_churn": will_churn,
    })


def _report(df=None, target="will_churn"):
    df = _panel() if df is None else df
    return analyze_target(df, infer_schema(df), target)


def test_named_binary_column_is_the_suggested_target():
    df = _panel()
    candidates = detect_targets(df, infer_schema(df))

    assert candidates[0].column == "will_churn"
    assert "is_premium" in [candidate.column for candidate in candidates]


def test_summary_and_imbalance():
    df = _panel()
    report = _report(df)

    assert report.kind == "binary" and report.positive_label == 1
    assert abs(report.positive_rate - df["will_churn"].mean()) < 1e-9
    assert report.imbalance.severity in {"warning", "critical"}


def test_leakage_is_flagged():
    report = _report()
    critical = {f.column for f in report.findings if f.severity == "critical"}
    warned = {f.column for f in report.findings if f.severity == "warning"}

    assert "refund_after_close" in critical          # separates classes perfectly
    assert "closed_on" in warned                      # dated after the snapshot month
    assert "complaints" not in critical


def test_real_signal_and_segment_are_ranked():
    report = _report()

    complaints = next(f for f in report.features if f.column == "complaints")
    footfall = next(f for f in report.features if f.column == "footfall")
    assert complaints.auc > 0.6 and complaints.direction.startswith("higher")
    assert footfall.strength < complaints.strength

    region = next(s for s in report.segments if s.column == "region")
    assert region.highest[0] == "Coast"
    assert region.p_value < 0.05


def test_panel_is_detected_and_split_is_by_time():
    df = _panel()
    report = _report(df)

    assert report.panel.entity == "store_code"
    assert (report.panel.n_entities, report.panel.n_periods) == (120, 18)

    split = report.split
    assert split.method == "time"
    train, test = split.table.iloc[0], split.table.iloc[1]
    assert train["To"] < test["From"]
    assert train["Rows"] + test["Rows"] == len(df)
    assert "store code" in split.summary.lower()


def test_ids_and_dates_are_not_features():
    excluded = {column for column, _ in _report().excluded}
    assert {"store_code", "period", "closed_on"} <= excluded


def test_text_target_and_no_date_column():
    df = _panel().drop(columns=["period", "closed_on"])
    df["will_churn"] = np.where(df["will_churn"] == 1, "Yes", "No")

    report = _report(df)

    assert report.positive_label == "Yes"
    assert report.split.method == "stratified"
    assert report.panel is None


def test_numeric_target():
    report = _report(target="footfall")

    assert report.kind == "numeric"
    assert report.features and all(f.correlation is not None for f in report.features)


def test_markdown_report_and_panel_render():
    df = _panel()
    report = _report(df)
    text = report_markdown(report, "stores.csv")

    for heading in ("# Target EDA report: will_churn", "## Warnings and checks",
                    "## Train / test split", "## Segments", "## Numeric features"):
        assert heading in text

    from ui import panels

    bundle = DatasetBundle(
        id="t", name="stores.csv", content=b"", sheets=[], sheet=None,
        df=df, schema=infer_schema(df),
    )
    json.dumps(panels.target_panel(bundle), cls=plotly.utils.PlotlyJSONEncoder)
