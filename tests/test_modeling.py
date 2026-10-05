"""
Phase 3a: goal-driven model builder.

Synthetic data with known signal and a planted leak, using names
unrelated to any project, so nothing here depends on one schema.
"""

import json
import time

import numpy as np
import plotly
import pytest

from core.modeling import build_model, list_models, load_model, propose_goal, save_model, target_options
from core.schema_inference import infer_schema
from tests.sample_data import churn_panel, retail_dataframe
from service.session import DatasetBundle


def _setup(df, goal, target):
    schema = infer_schema(df)
    return schema, propose_goal(df, schema, goal, target)


@pytest.fixture(scope="module")
def churn_result():
    df = churn_panel()
    schema, spec = _setup(df, "rank", "will_churn")
    return df, spec, build_model(df, schema, spec)


def test_targets_are_offered_by_goal_type():
    df = churn_panel()
    schema = infer_schema(df)

    assert target_options(df, schema, "classify")[0] == "will_churn"
    assert "footfall" in target_options(df, schema, "regress")
    assert "will_churn" not in target_options(df, schema, "regress")


def test_wrong_target_type_is_rejected():
    df = churn_panel()
    with pytest.raises(ValueError):
        _setup(df, "regress", "region")


def test_goal_excludes_ids_dates_and_leaks():
    df = churn_panel()
    _, spec = _setup(df, "classify", "will_churn")
    excluded = dict(spec.excluded)

    assert {"store_code", "period", "closed_on", "refund_after_close"} <= set(excluded)
    assert "leakage" in excluded["refund_after_close"]
    assert "complaints" in spec.features and "region" in spec.features
    assert spec.time_column == "period" and spec.entity == "store_code"


def test_classifier_beats_baseline_on_a_time_split(churn_result):
    df, spec, result = churn_result

    assert result.best_name != "Baseline (no features)"
    assert result.test_metrics["pr_auc"] > result.baseline_metrics["pr_auc"] * 1.5
    assert result.test_metrics["roc_auc"] > 0.65

    split = result.split
    assert split.method == "time"
    assert df.loc[split.train, "period"].max() < df.loc[split.test, "period"].min()
    assert df.loc[split.inner_train, "period"].max() < df.loc[split.validation, "period"].min()


def test_explanations_and_scores(churn_result):
    df, spec, result = churn_result

    assert result.importance.iloc[0]["Feature"] in {"complaints", "region"}

    latest = df["period"].max()
    assert len(result.scored) == (df["period"] == latest).sum()
    assert list(result.scored.columns[:3]) == ["Rank", "store_code", "Probability"]
    assert result.scored["Probability"].is_monotonic_decreasing
    assert result.scored["Reasons"].str.len().gt(0).mean() > 0.5

    assert result.lift is not None and result.lift["Cumulative capture"].iloc[-1] == 1.0


def test_regression_beats_baseline():
    df = retail_dataframe(1500)
    schema, spec = _setup(df, "regress", "Sales")
    result = build_model(df, schema, spec)

    assert result.task == "regression"
    assert result.test_metrics["mae"] < result.baseline_metrics["mae"] * 0.7
    assert "Predicted Sales" in result.scored.columns


def test_undated_data_uses_a_stratified_split():
    df = churn_panel().drop(columns=["period", "closed_on"])
    schema, spec = _setup(df, "classify", "will_churn")
    result = build_model(df, schema, spec)

    assert result.split.method == "stratified"
    rate = df["will_churn"].mean()
    assert abs(df.loc[result.split.test, "will_churn"].mean() - rate) < 0.02


def test_registry_round_trip(churn_result, tmp_path):
    df, spec, result = churn_result

    meta = save_model(result, df, "stores.csv", folder=tmp_path)
    assert [m["name"] for m in list_models(tmp_path)] == [meta["name"]]

    loaded = load_model(meta["name"], folder=tmp_path)
    assert loaded["meta"]["goal"]["target"] == "will_churn"

    from core.modeling.features import prepare_frame

    sample = df.head(50)
    X = prepare_frame(sample, loaded["numeric"], loaded["categorical"])
    assert np.allclose(loaded["pipeline"].predict_proba(X)[:, 1], result.predict(sample))


def test_background_job_and_panels():
    from ui import panels

    df = churn_panel(rows_per_store=12, stores=80)
    bundle = DatasetBundle(
        id="m", name="stores.csv", content=b"", sheets=[], sheet=None,
        df=df, schema=infer_schema(df),
    )

    json.dumps(panels.model_panel(bundle), cls=plotly.utils.PlotlyJSONEncoder)

    job_id = bundle.start_model_job("rank", "will_churn")
    deadline = time.time() + 120
    while bundle.model_job(job_id)["status"] == "running" and time.time() < deadline:
        time.sleep(0.5)

    job = bundle.model_job(job_id)
    assert job["status"] == "done", job["error"]
    assert job["messages"][-1] == "Done"
    assert bundle.latest_model is job["result"]

    json.dumps(panels.model_results_view(job["result"]), cls=plotly.utils.PlotlyJSONEncoder)
