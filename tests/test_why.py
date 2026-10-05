"""
Phase 4a: why did a number change?

Synthetic sales with a planted collapse in the last month (one region,
one product, one rep), a pure mix shift, and names unrelated to any project.
"""

import json
import time

import numpy as np
import pandas as pd
import plotly

from core.forecast import ROW_COUNT
from core.schema_inference import infer_schema
from core.why import ChangeSpec, drill_options, investigate
from core.why.change import breakdown, choose_periods, explain_change
from tests.sample_data import retail_dataframe
from service.session import DatasetBundle

REPS = {"North": ["R1", "R2"], "South": ["R3", "R4"], "West": ["R5", "R6", "R7"]}


def _sales(seed: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for month in pd.date_range("2024-01-01", periods=12, freq="MS"):
        for region, reps in REPS.items():
            for rep in reps:
                for product in ["Gadget", "Widget", "Gizmo"]:
                    for _ in range(4):
                        amount = rng.normal(100, 8)
                        # Planted: in the last month R7 sells almost no Gadgets.
                        if month == pd.Timestamp("2024-12-01") and rep == "R7" and product == "Gadget":
                            amount *= 0.02
                        day = month + pd.Timedelta(days=int(rng.integers(0, 27)))
                        rows.append({"sold_on": day, "region": region, "rep_code": rep,
                                     "rep_name": f"Rep {rep}", "product": product,
                                     "amount": round(amount, 2), "logged_at": "09:30"})
    return pd.DataFrame(rows)


def _spec(frame, **overrides):
    schema = infer_schema(frame)
    values = dict(time="sold_on", measure="amount", aggregation="sum", freq="M",
                  dimensions=drill_options(frame, schema))
    values.update(overrides)
    return ChangeSpec(**values)


def test_drill_options_skip_duplicates_and_times_of_day():
    frame = _sales()
    options = drill_options(frame, infer_schema(frame))

    assert {"region", "product"} <= set(options)
    assert ("rep_code" in options) != ("rep_name" in options)   # one-to-one: only one kept
    assert "logged_at" not in options


def test_latest_month_is_compared_with_the_previous_one():
    frame = _sales()
    current, previous = choose_periods(frame, _spec(frame))
    assert (current, previous) == (pd.Timestamp("2024-12-01"), pd.Timestamp("2024-11-01"))

    year = choose_periods(_sales_two_years(), _spec(_sales_two_years(), compare="year"))
    assert year == (pd.Timestamp("2025-12-01"), pd.Timestamp("2024-12-01"))


def _sales_two_years() -> pd.DataFrame:
    frame = _sales()
    later = frame.copy()
    later["sold_on"] = later["sold_on"] + pd.DateOffset(years=1)
    return pd.concat([frame, later], ignore_index=True)


def test_drill_down_finds_the_planted_cause():
    frame = _sales()
    steps = explain_change(frame, _spec(frame))

    assert steps[0].comparison.change < 0 and "fell" in steps[0].text
    # The chain may take the dimensions in any order; it must end at the cause.
    path = dict(steps[-1].path)
    assert path.get("region") == "West" and path.get("product") == "Gadget"
    assert path.get("rep_code", path.get("rep_name")) in ("R7", "Rep R7")
    assert steps[-1].comparison.change_pct < -0.9


def test_contributions_add_up_including_missing_values():
    frame = _sales()
    frame.loc[frame.sample(frac=0.1, random_state=2).index, "product"] = None
    spec = _spec(frame)
    current, previous = choose_periods(frame, spec)
    table = breakdown(frame, spec, "product", current, previous).table

    month = frame["sold_on"].dt.to_period("M").dt.start_time
    total_change = frame.loc[month == current, "amount"].sum() - frame.loc[month == previous, "amount"].sum()
    assert abs(table["Change"].sum() - total_change) < 1e-6
    assert "(missing)" in set(table["product"])


def test_mix_and_rate_separate_a_pure_mix_shift():
    rows = []
    for month, premium_share in [("2024-01-01", 0.2), ("2024-02-01", 0.2), ("2024-03-01", 0.6)]:
        for i in range(100):
            tier = "Premium" if i < premium_share * 100 else "Basic"
            rows.append({"sold_on": pd.Timestamp(month) + pd.Timedelta(days=i % 27), "tier": tier,
                         "price": 50.0 if tier == "Premium" else 10.0})
    frame = pd.DataFrame(rows)
    spec = ChangeSpec(time="sold_on", measure="price", aggregation="mean", freq="M", dimensions=["tier"])
    current, previous = choose_periods(frame, spec)
    table = breakdown(frame, spec, "tier", current, previous).table

    assert abs(table["Rate effect"].sum()) < 1e-9          # prices didn't change
    assert abs(table["Mix effect"].sum() - 16.0) < 1e-9     # average 18 -> 34 from the mix only


def test_attention_and_unusual_flag_the_collapsing_rep():
    frame = _sales()
    result = investigate(frame, _spec(frame), attention_dimension="rep_code")

    assert result.attention.iloc[0]["rep_code"] == "R7"
    assert "down" in result.attention.iloc[0]["Why"]
    assert "R7" in set(result.unusual.iloc[:, 0])


def test_ask_box_answers_why_questions():
    frame = retail_dataframe(3000)
    bundle = DatasetBundle(id="w", name="retail.csv", content=b"", sheets=[], sheet=None,
                           df=frame, schema=infer_schema(frame))

    entry = bundle.ask("why did Sales change")
    assert entry["error"] is None
    assert "Sales" in entry["answer"] and ("rose" in entry["answer"] or "fell" in entry["answer"])
    assert entry["table"] is not None

    counts = bundle.ask("why did the number of customers drop")
    assert counts["error"] is None and "Number of rows" in counts["answer"]


def test_dashboard_tab_and_background_job():
    from ui import panels

    frame = _sales()
    bundle = DatasetBundle(id="v", name="sales.csv", content=b"", sheets=[], sheet=None,
                           df=frame, schema=infer_schema(frame))
    json.dumps(panels.investigate_panel(bundle), cls=plotly.utils.PlotlyJSONEncoder)

    job_id = bundle.start_why_job(None, _spec(frame), "rep_code", True)
    deadline = time.time() + 60
    while bundle.model_job(job_id)["status"] == "running" and time.time() < deadline:
        time.sleep(0.2)

    job = bundle.model_job(job_id)
    assert job["status"] == "done", job["error"]
    json.dumps(panels.investigate_results_view(job["result"]), cls=plotly.utils.PlotlyJSONEncoder)
