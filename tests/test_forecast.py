"""
Phase 3c: forecasting.

Synthetic order rows whose monthly totals follow a known trend and
yearly seasonality, with names unrelated to any project.
"""

import json
import time

import numpy as np
import pandas as pd
import plotly
import pytest

from core.forecast import (
    ROW_COUNT,
    TOTAL,
    SeriesSpec,
    build_forecast,
    build_series,
    measure_options,
    suggest_freq,
)
from core.schema_inference import infer_schema
from service.session import DatasetBundle

REGIONS = {"North": 1.0, "South": 0.6, "Lakes": 0.4}


def _orders(months: int = 72, seed: int = 3, end_mid_month: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    starts = pd.date_range("2019-01-01", periods=months, freq="MS")
    rows = []

    for t, start in enumerate(starts):
        level = 200 + 3.0 * t + 40 * np.sin(2 * np.pi * start.month / 12)
        for region, share in REGIONS.items():
            total = max(level * share + rng.normal(0, 4), 1)
            for part in rng.dirichlet(np.ones(6)) * total:
                day = start + pd.Timedelta(days=int(rng.integers(0, 28)))
                rows.append({"order_date": day, "region": region, "amount": round(part, 2),
                             "store_rating": {"North": 4.1, "South": 3.7, "Lakes": 4.4}[region]})

    frame = pd.DataFrame(rows)
    if end_mid_month:
        last = starts[-1]
        frame = frame[frame["order_date"] < last + pd.Timedelta(days=10)]
    return frame


@pytest.fixture(scope="module")
def forecast():
    frame = _orders()
    return frame, build_forecast(frame, SeriesSpec("order_date", "amount", "sum", "region", "M", 3))


def test_series_are_bucketed_split_and_consistent():
    frame = _orders(months=30)
    built = build_series(frame, SeriesSpec("order_date", "amount", "sum", "region", "M", 3))

    assert set(built.series) == {TOTAL, *REGIONS}
    assert len(built.series[TOTAL]) == 30
    parts = sum(built.series[region] for region in REGIONS)
    assert np.allclose(parts, built.series[TOTAL])

    counts = build_series(frame, SeriesSpec("order_date", ROW_COUNT, "count", None, "M", 3))
    assert (counts.series[TOTAL] == 18).all()   # 3 regions x 6 orders per month


def test_incomplete_last_period_is_dropped():
    built = build_series(_orders(months=30, end_mid_month=True), SeriesSpec("order_date", "amount", "sum", None, "M", 3))

    assert len(built.series[TOTAL]) == 29
    assert built.dropped_last is not None and "incomplete" in built.notes[0]


def test_measure_and_frequency_defaults():
    frame = _orders(months=30)
    options = measure_options(frame, infer_schema(frame))

    # store_rating is constant per region: an attribute, listed after the row count.
    assert options[0] == "amount"
    assert options.index(ROW_COUNT) < options.index("store_rating")

    days = pd.Series(pd.date_range("2024-01-01", periods=60, freq="D"))
    assert suggest_freq(days) == "D"
    assert suggest_freq(pd.Series(pd.date_range("2024-01-01", periods=200, freq="D"))) == "W"
    assert suggest_freq(pd.Series(pd.date_range("2023-01-01", periods=700, freq="D"))) == "M"
    assert suggest_freq(pd.Series(pd.date_range("2023-01-01", periods=20, freq="MS"))) == "M"


def test_models_beat_baselines_and_detect_structure(forecast):
    _, result = forecast
    table = result.metrics.set_index("Model")

    best = table.loc[result.best_model, "MASE"]
    baselines = table[table["Type"] == "baseline"]["MASE"]
    assert result.best_model not in baselines.index
    assert best < baselines.min()

    assert result.diagnostics["seasonal_strength"] > 0.5
    assert result.diagnostics["trend_strength"] > 0.5
    assert result.decomposition is not None


def test_forecast_shape_and_interval_coverage():
    full = _orders()
    cutoff = pd.Timestamp("2024-10-01")
    result = build_forecast(full[full["order_date"] < cutoff], SeriesSpec("order_date", "amount", "sum", None, "M", 3))

    total = result.forecasts[result.forecasts["Series"] == TOTAL]
    assert list(total["Period"]) == list(pd.date_range(cutoff, periods=3, freq="MS"))
    assert (total["Low 95%"] <= total["Forecast"]).all() and (total["Forecast"] <= total["High 95%"]).all()

    actual = full[full["order_date"] >= cutoff].groupby(full["order_date"].dt.to_period("M").dt.start_time)["amount"].sum()
    inside = ((actual.to_numpy() >= total["Low 95%"].to_numpy()) & (actual.to_numpy() <= total["High 95%"].to_numpy())).sum()
    assert inside >= 2


def test_too_long_horizon_is_refused():
    with pytest.raises(ValueError):
        build_forecast(_orders(months=12), SeriesSpec("order_date", "amount", "sum", None, "M", 8))


def test_dashboard_tab_and_background_job():
    from ui import panels

    frame = _orders(months=36)
    bundle = DatasetBundle(id="f", name="orders.csv", content=b"", sheets=[], sheet=None,
                           df=frame, schema=infer_schema(frame))

    json.dumps(panels.forecast_panel(bundle), cls=plotly.utils.PlotlyJSONEncoder)

    job_id = bundle.start_forecast_job(None, SeriesSpec("order_date", "amount", "sum", "region", "M", 3))
    deadline = time.time() + 120
    while bundle.model_job(job_id)["status"] == "running" and time.time() < deadline:
        time.sleep(0.5)

    job = bundle.model_job(job_id)
    assert job["status"] == "done", job["error"]

    json.dumps(panels.forecast_results_view(job["result"]), cls=plotly.utils.PlotlyJSONEncoder)
    json.dumps(panels.forecast_chart(job["result"], "South"), cls=plotly.utils.PlotlyJSONEncoder)
