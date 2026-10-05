"""
Phase 6: free-text goal box (rules + language model -> validated plan) and the LLM layer.

Synthetic data only: a sales workbook (contacts / reps / accounts), a
retail table, a store panel and a regional orders series. The language
model is a scripted stand-in, so tests never need a running model.
"""

import time

import pytest
from fastapi.testclient import TestClient

from core.intent import IntentSpec, complete, interpret, read_goal
from core.llm import LLMError, ScriptedProvider, get_provider, provider_status, reset_provider, structured, use_provider
from core.llm.provider import OllamaProvider
from service.session import DatasetStore
from tests.sample_data import churn_panel, retail_dataframe
from tests.test_forecast import _orders
from tests.test_recommend import _workbook


@pytest.fixture(scope="module", autouse=True)
def no_model():
    use_provider(None)
    yield
    reset_provider()


@pytest.fixture(scope="module")
def store():
    return DatasetStore(capacity=8)


@pytest.fixture(scope="module")
def sales(store):
    return store.load(_workbook(), "sales.xlsx", "Contacts")


@pytest.fixture(scope="module")
def retail(store):
    return store.load(retail_dataframe(1500).to_csv(index=False).encode(), "retail.csv")


@pytest.fixture(scope="module")
def stores(store):
    return store.load(churn_panel(rows_per_store=12, stores=120).to_csv(index=False).encode(), "stores.csv")


def _wait(bundle, job_id, timeout=180):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = bundle.model_job(job_id)
        if job["status"] != "running":
            return job
        time.sleep(0.3)
    raise AssertionError("job did not finish")


# ----------------------------------------------------------------------
# Rules
# ----------------------------------------------------------------------

def test_plan_goal_read_by_rules(sales):
    plan = sales.interpret_goal("plan which accounts each rep should contact next month")
    spec = plan.spec
    assert plan.source == "rules" and plan.ready
    assert (spec.task, spec.user, spec.item, spec.period) == ("recommend", "RepCode", "AccountKey", "month")
    assert spec.sheet == "Contacts" and spec.outcome == "Result"
    assert any("Success means" in a for a in plan.assumptions)


def test_unknown_words_are_reported_not_guessed(sales):
    plan = sales.interpret_goal("plan calls to clients for each salesperson next week")
    assert plan.spec.task == "recommend" and plan.spec.period == "days" and plan.spec.days == 5
    warning = " ".join(plan.warnings)
    assert "'clients'" in warning and "'salesperson'" in warning
    assert "language model" in warning.lower()


def test_forecast_goal_with_split_and_horizon(retail):
    plan = retail.interpret_goal("forecast sales by region for the next 6 months")
    spec = plan.spec
    assert plan.ready and (spec.task, spec.measure, spec.group, spec.freq, spec.horizon) == ("forecast", "Sales", "Region", "M", 6)
    assert "next 6 months" in plan.summary


def test_why_rank_segment_and_ask_goals(retail, stores):
    why = retail.interpret_goal("why did profit drop")
    assert (why.spec.task, why.spec.measure) == ("why", "Profit") and why.ready

    rank = stores.interpret_goal("which stores are most likely to churn")
    assert (rank.spec.task, rank.spec.target) == ("rank", "will_churn") and rank.ready

    segment = retail.interpret_goal("find unusual customers")
    assert (segment.spec.task, segment.spec.unit) == ("segment", "customer_id")

    ask = retail.interpret_goal("list customers in North region")
    assert ask.spec.task == "ask" and ask.spec.question == "list customers in North region"


def test_vague_goal_asks_what_to_do(retail):
    plan = retail.interpret_goal("make something nice")
    assert not plan.ready and plan.questions[0].field == "task"
    assert "forecast" in plan.questions[0].options


def test_rule_reading_does_not_use_unique_keys_as_roles(sales):
    reading = read_goal("plan contacts of each rep to accounts", sales.goal_view())
    assert reading.fields["user"] == "RepCode" and reading.fields["item"] == "AccountKey"


# ----------------------------------------------------------------------
# Validation (whoever proposed the spec)
# ----------------------------------------------------------------------

def test_invented_columns_become_questions(retail):
    plan = retail.check_goal({"task": "forecast", "measure": "Turnover"})
    assert not plan.ready
    question = next(q for q in plan.questions if q.field == "measure")
    assert "Turnover" in question.text and "Sales" in question.options

    fixed = retail.check_goal({"task": "forecast", "measure": "sales"})
    assert fixed.ready and fixed.spec.measure == "Sales"
    assert any("Read 'sales' as the column 'Sales'" in a for a in fixed.assumptions)


def test_role_mismatch_and_other_table(retail, sales):
    wrong = retail.check_goal({"task": "rank", "target": "Region"})
    assert not wrong.ready and "doesn't fit" in wrong.questions[0].text

    # Model goals use the loaded table; a target from another table is explained.
    other = sales.check_goal({"task": "regress", "target": "Tier", "sheet": "Accounts"})
    assert any("loaded table" in w for w in other.warnings)


def test_defaults_are_listed_as_assumptions(retail):
    plan = retail.check_goal({"task": "forecast"})
    assert plan.ready and plan.spec.time == "order_date"
    assert any("chosen automatically" in a for a in plan.assumptions)
    assert "measure" in plan.options and "group" in plan.options


def test_suggestions_are_ready_goals(sales):
    suggestions = sales.goal_suggestions()
    tasks = [plan.spec.task for plan in suggestions]
    assert "recommend" in tasks and "forecast" in tasks
    assert all(plan.ready for plan in suggestions)
    recommend = next(p for p in suggestions if p.spec.task == "recommend")
    assert recommend.title == "Plan which accounts each rep should contact"


# ----------------------------------------------------------------------
# Language model (scripted)
# ----------------------------------------------------------------------

def test_model_maps_synonyms_when_rules_are_unsure(sales):
    provider = ScriptedProvider([{"task": "recommend", "user": "RepCode", "item": "AccountKey",
                                  "period": "days", "days": 5, "unknown_terms": []}])
    plan = interpret("plan calls to clients for each salesperson next week", sales.goal_view(), provider)
    assert plan.source == "language model" and plan.ready
    assert (plan.spec.user, plan.spec.item, plan.spec.days) == ("RepCode", "AccountKey", 5)
    assert not plan.warnings
    prompt = provider.calls[0][1]["content"]
    assert "RepCode" in prompt and "Contacts" in prompt and "Goal: plan calls" in prompt


def test_model_output_is_validated(retail):
    provider = ScriptedProvider([{"task": "forecast", "measure": "Turnover", "unknown_terms": ["turnover"]}])
    plan = interpret("project our turnover", retail.goal_view(), provider)
    assert not plan.ready and any(q.field == "measure" for q in plan.questions)
    assert any("turnover" in w for w in plan.warnings)


def test_rules_skip_the_model_when_sure(retail):
    provider = ScriptedProvider([])
    plan = interpret("forecast sales by region for the next 6 months", retail.goal_view(), provider)
    assert plan.source == "rules" and not provider.calls


def test_model_failure_falls_back_to_rules(sales):
    provider = ScriptedProvider(["not json", "still not json"])
    plan = interpret("plan calls to clients for each salesperson next week", sales.goal_view(), provider)
    assert plan.source == "rules" and plan.llm_error and plan.spec.task == "recommend"


def test_structured_output_retries_once():
    from pydantic import BaseModel

    class Answer(BaseModel):
        value: int

    provider = ScriptedProvider(['{"value": "many"}', '```json\n{"value": 3}\n```'])
    assert structured(provider, "system", "user", Answer).value == 3
    assert "not valid" in provider.calls[1][-1]["content"]

    with pytest.raises(LLMError):
        structured(ScriptedProvider(["{}", "{}"]), "system", "user", Answer)


def test_provider_settings(monkeypatch):
    reset_provider()
    monkeypatch.setenv("AIDS_LLM_PROVIDER", "none")
    assert get_provider(refresh=True) is None
    assert "switched off" in provider_status()["detail"]

    monkeypatch.setenv("AIDS_LLM_PROVIDER", "ollama")
    monkeypatch.setenv("AIDS_OLLAMA_HOST", "http://127.0.0.1:9")
    assert get_provider(refresh=True) is None
    assert "not running" in provider_status()["detail"]
    assert OllamaProvider().model == "qwen3.5:4b"
    use_provider(None)


# ----------------------------------------------------------------------
# Running goals
# ----------------------------------------------------------------------

def test_run_goals_start_the_pipelines(retail):
    started = retail.run_goal(retail.interpret_goal("forecast sales by region for the next 3 months").spec)
    job = _wait(retail, started["job_id"])
    assert started["tab"] == "forecast" and job["status"] == "done", job["error"]
    assert job["result"].spec.group == "Region"

    started = retail.run_goal({"task": "why", "measure": "Profit"})
    job = _wait(retail, started["job_id"])
    assert job["status"] == "done", job["error"]
    assert retail.latest_why is not None

    answered = retail.run_goal({"task": "ask", "question": "how many customers are in North region"})
    assert answered["tab"] == "ask" and answered["entry"]["error"] is None

    with pytest.raises(ValueError, match="not complete"):
        retail.run_goal({"task": "forecast", "measure": "Turnover"})


def test_goal_api(sales):
    from api.main import app, store as api_store

    client = TestClient(app)
    api_store._items[sales.id] = sales

    assert client.get("/llm/status").json()["available"] is False

    plan = client.post(f"/datasets/{sales.id}/goals/interpret",
                       json={"text": "plan which accounts each rep should contact next month"}).json()
    assert plan["ready"] and plan["spec"]["user"] == "RepCode" and plan["tab"] == "recommend"

    checked = client.post(f"/datasets/{sales.id}/goals/check", json={"task": "forecast", "measure": "Turnover"}).json()
    assert not checked["ready"] and checked["questions"][0]["field"] == "measure"

    suggestions = client.get(f"/datasets/{sales.id}/goals/suggestions").json()
    assert all(item["ready"] for item in suggestions)

    run = client.post(f"/datasets/{sales.id}/goals/run", json={"task": "forecast", "sheet": "Contacts"}).json()
    assert run["tab"] == "forecast" and run["status_url"].endswith(run["job_id"])
    assert client.post(f"/datasets/{sales.id}/goals/run", json={"task": "forecast", "measure": "Turnover"}).status_code == 400
