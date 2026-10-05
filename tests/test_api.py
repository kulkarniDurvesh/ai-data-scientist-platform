"""
Phase 5: HTTP API.

Exercises every endpoint through FastAPI's test client with synthetic
datasets (retail, a store panel, a sales workbook); no pharma data.
"""

import io
import time

import pytest
from fastapi.testclient import TestClient

from api.main import app
from tests.sample_data import churn_panel, retail_dataframe
from tests.test_recommend import _workbook
from tests.test_forecast import _orders

client = TestClient(app)


def _upload(frame, name="data.csv"):
    response = client.post("/datasets", files={"file": (name, frame.to_csv(index=False).encode(), "text/csv")})
    assert response.status_code == 200, response.text
    return response.json()


def _wait(dataset_id, job_id, timeout=180):
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = client.get(f"/datasets/{dataset_id}/jobs/{job_id}").json()
        if status["status"] != "running":
            return status
        time.sleep(0.5)
    raise AssertionError("job did not finish")


@pytest.fixture(scope="module")
def retail():
    return _upload(retail_dataframe(1500), "retail.csv")


def test_health_and_openapi():
    assert client.get("/health").json()["status"] == "ok"
    schema = client.get("/openapi.json").json()
    assert "/datasets/{dataset_id}/ask" in schema["paths"]
    assert "AskResponse" in schema["components"]["schemas"]


def test_dataset_schema_quality_insights(retail):
    info = client.get(f"/datasets/{retail['id']}").json()
    assert info["rows"] == 1500 and info["primary_time"] == "order_date"
    roles = {c["column"]: c["role"] for c in info["column_info"]}
    assert roles["Sales"] == "measure" and roles["customer_id"] == "identifier"

    issues = client.get(f"/datasets/{retail['id']}/quality").json()
    assert any(issue["column"] == "days_to_ship" for issue in issues)
    assert client.get(f"/datasets/{retail['id']}/insights?top=3").json()

    assert client.get("/datasets/nope").status_code == 404


def test_ask(retail):
    answer = client.post(f"/datasets/{retail['id']}/ask", json={"question": "Which customers are in North region"}).json()
    assert answer["error"] is None and answer["answer"].startswith("Found")
    assert answer["table"]["total_rows"] > 0

    chart = client.post(f"/datasets/{retail['id']}/ask", json={"question": "pie chart of customers by Category"}).json()
    assert chart["chart"]["chart_type"] == "PIE"

    why = client.post(f"/datasets/{retail['id']}/ask", json={"question": "why did Sales change"}).json()
    assert "Sales" in why["answer"]


def test_target_model_save_and_score():
    panel = churn_panel(rows_per_store=12, stores=120)
    dataset = _upload(panel, "stores.csv")

    report = client.get(f"/datasets/{dataset['id']}/target").json()
    assert report["target"] == "will_churn" and report["split_method"] == "time"
    assert "# Target EDA report" in report["markdown"]

    job = client.post(f"/datasets/{dataset['id']}/models", json={"goal_type": "rank"}).json()
    status = _wait(dataset["id"], job["job_id"])
    assert status["status"] == "done", status["error"]
    result = status["result"]
    assert result["type"] == "model" and result["best_model"] != "Baseline (no features)"
    assert result["test_metrics"]["pr_auc"] > result["baseline_metrics"]["pr_auc"]

    saved = client.post(f"/datasets/{dataset['id']}/models/save").json()
    assert any(m["name"] == saved["name"] for m in client.get("/models").json())

    rows = panel.head(5).drop(columns=["will_churn"]).astype(str).to_dict("records")
    for row in rows:
        row["complaints"] = float(row["complaints"])
        row["footfall"] = float(row["footfall"])
        row["is_premium"] = float(row["is_premium"])
    scores = client.post(f"/models/{saved['name']}/score", json={"rows": rows}).json()
    assert len(scores["predictions"]) == 5 and all(0 <= p <= 1 for p in scores["predictions"])

    bad = client.post(f"/models/{saved['name']}/score", json={"rows": [{"complaints": 1}]})
    assert bad.status_code == 400 and "missing columns" in bad.json()["detail"]

    import shutil
    from core.modeling.registry import DEFAULT_FOLDER
    shutil.rmtree(DEFAULT_FOLDER / saved["name"], ignore_errors=True)


def test_recommend_on_a_workbook():
    response = client.post("/datasets", files={"file": ("sales.xlsx", _workbook(), "application/octet-stream")},
                           data={"sheet": "Accounts"})
    dataset = response.json()

    job = client.post(f"/datasets/{dataset['id']}/recommend", json={"period": "days", "days": 3, "capacity": 2}).json()
    status = _wait(dataset["id"], job["job_id"])
    assert status["status"] == "done", status["error"]
    result = status["result"]
    assert result["roles"]["user"] == "RepCode" and result["plan"]["total_rows"] > 0

    one = client.get(f"/datasets/{dataset['id']}/recommend/latest", params={"user": "RP01"}).json()
    assert {row["RepCode"] for row in one["plan"]["rows"]} <= {"RP01"}


def test_forecast_segments_why():
    dataset = _upload(_orders(months=36), "orders.csv")

    job = client.post(f"/datasets/{dataset['id']}/forecast", json={"group": "region"}).json()
    status = _wait(dataset["id"], job["job_id"])
    assert status["status"] == "done", status["error"]
    assert status["result"]["spec"]["measure"] == "amount"
    assert status["result"]["forecasts"]["total_rows"] == 4 * 3

    job = client.post(f"/datasets/{dataset['id']}/segments", json={"features": ["amount", "region"]}).json()
    status = _wait(dataset["id"], job["job_id"])
    assert status["status"] == "done", status["error"]
    assert status["result"]["segments"] >= 2

    why = client.post(f"/datasets/{dataset['id']}/why", json={"dimensions": ["region"], "attention": "none"}).json()
    assert why["type"] == "investigation" and why["explanation"][0]["text"]


def test_kpis_from_inline_yaml(retail):
    domain = """
name: Retail test
roles:
  customer: customer_id
  date: order_date
kpis:
  - id: revenue
    value: {sum: Sales}
  - id: return_rate
    numerator: {count: rows, where: {Returned: "Yes"}}
    denominator: {count: rows}
    direction: down
"""
    report = client.post(f"/datasets/{retail['id']}/kpis", json={"domain_yaml": domain}).json()
    labels = [row["KPI"] for row in report["kpis"]["rows"]]
    assert labels == ["Revenue", "Return Rate"] and report["unavailable"]["total_rows"] == 0

    starter = client.get(f"/datasets/{retail['id']}/kpis/starter").json()
    assert "kpis:" in starter["yaml"]

    assert client.post(f"/datasets/{retail['id']}/kpis", json={"domain": "no_such_file"}).status_code == 404


def test_api_key(monkeypatch, retail):
    monkeypatch.setenv("AIDS_API_KEY", "secret")
    assert client.get("/health").status_code == 401
    assert client.get("/health", headers={"X-API-Key": "secret"}).status_code == 200
