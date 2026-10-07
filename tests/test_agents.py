"""
Phase 8a: tool-calling agents.

Synthetic retail data and the platform's document folder (local LSA
vectors); the language model is a scripted stand-in that plays the steps.
"""

import time

import pytest
from fastapi.testclient import TestClient

from core.agents import Agent, NoArguments, Tool, run_agent
from core.llm import ScriptedProvider, reset_provider, use_provider
from service.agents import conversation_runs, make_agent, start_agent_job
from service.session import DatasetStore
from tests.sample_data import retail_dataframe


@pytest.fixture(autouse=True)
def setup(monkeypatch):
    monkeypatch.setenv("AIDS_EMBED_PROVIDER", "lsa")
    use_provider(None)
    yield
    use_provider(None)


@pytest.fixture(scope="module", autouse=True)
def reset_after():
    yield
    reset_provider()


@pytest.fixture(scope="module")
def frame():
    return retail_dataframe(800)


@pytest.fixture(scope="module")
def retail(frame):
    return DatasetStore().load(frame.to_csv(index=False).encode(), "retail.csv")


def _wait(bundle, job_id):
    while bundle.model_job(job_id)["status"] == "running":
        time.sleep(0.2)
    return bundle.model_job(job_id)


def test_automl_agent_interprets_runs_and_answers(retail):
    provider = ScriptedProvider([
        {"thought": "read the goal", "action": "interpret_goal", "arguments": {"text": "forecast sales by region for the next 3 months"}},
        {"thought": "ready", "action": "run_goal", "arguments": {"spec": {"task": "forecast", "measure": "Sales", "group": "Region", "freq": "M", "horizon": 3}}},
        {"thought": "explain", "action": "final_answer", "answer": "The forecast for total Sales by Region is ready in the Forecast tab."},
    ])
    run = run_agent(make_agent(retail, "automl"), "forecast sales by region for the next 3 months", provider)
    assert run.status == "done" and run.note is None
    assert [s.action for s in run.steps] == ["interpret_goal", "run_goal", "final_answer"]
    assert '"ready": true' in run.steps[0].observation
    assert '"tab": "forecast"' in run.steps[1].observation and "MASE" in run.steps[1].observation
    assert retail.latest_forecast.spec.group == "Region"
    # The prompt lists the agent's tools with their arguments.
    assert "interpret_goal(text: string" in provider.calls[0][0]["content"]


def test_agent_asks_the_user_and_continues_the_conversation(retail):
    provider = ScriptedProvider([
        {"thought": "unclear", "action": "interpret_goal", "arguments": {"text": "predict something"}},
        {"thought": "need the target", "action": "ask_user", "arguments": {"question": "Which column should be predicted: Profit or Sales?"}},
        {"thought": "user answered", "action": "interpret_goal", "arguments": {"text": "predict Profit"}},
        {"thought": "done", "action": "final_answer", "answer": "The plan predicts Profit; say run to start it."},
    ])
    use_provider(provider)
    job, conversation_id = start_agent_job(retail, "automl", "predict something")
    first = _wait(retail, job)["result"]
    assert first.status == "needs_input" and first.answer.startswith("Which column")
    # Tools never call the model themselves: one model call per agent step.
    assert len(provider.calls) == 2

    job, same = start_agent_job(retail, "automl", "Profit", conversation_id)
    second = _wait(retail, job)["result"]
    assert same == conversation_id and second.status == "done"
    runs = conversation_runs(retail, conversation_id)
    assert [r.message for r in runs] == ["predict something", "Profit"]


def test_analyst_combines_data_and_documents(retail, frame):
    provider = ScriptedProvider([
        {"thought": "compute", "action": "ask_data", "arguments": {"question": "how many customers are in North region"}},
        {"thought": "policy", "action": "ask_documents", "arguments": {"question": "what is the meal allowance on field days"}},
        {"thought": "answer", "action": "final_answer", "answer": "There are {n} customers in North; the meal allowance is INR 450 per day."},
    ])
    expected = frame.loc[frame["Region"] == "North", "customer_id"].nunique()
    provider.replies[2]["answer"] = provider.replies[2]["answer"].format(n=expected)
    run = run_agent(make_agent(retail, "analyst"), "customers in North, and the meal allowance?", provider)
    assert run.status == "done" and run.note is None
    assert f"There are {expected} customers" in run.steps[0].observation
    assert "INR 450" in run.steps[1].observation and "[1]" in run.steps[1].observation


def test_invented_numbers_in_the_final_answer_are_flagged(retail):
    provider = ScriptedProvider([
        {"thought": "compute", "action": "ask_data", "arguments": {"question": "how many customers are in North region"}},
        {"thought": "answer", "action": "final_answer", "answer": "There are 9999 customers in North."},
    ])
    run = run_agent(make_agent(retail, "analyst"), "customers in North?", provider)
    assert run.note and "9999" in run.note


def test_rounded_numbers_are_not_invented():
    from core.agents import Step, unsupported_numbers

    steps = [Step("", "run_goal", {}, observation='{"forecast": 43816.274, "rate": 0.4567}')]
    assert unsupported_numbers("About 43,816 (rate 0.46, exactly 43816.27).", steps, "") == []
    assert unsupported_numbers("About 43,900.", steps, "") == ["43,900"]


def test_bad_calls_become_observations_and_steps_are_limited(retail):
    calls = [{"thought": "oops", "action": "ask_data", "arguments": {}}] * 3
    run = run_agent(make_agent(retail, "analyst"), "anything", ScriptedProvider(calls), max_steps=3)
    assert run.status == "step_limit"
    assert all(not s.ok and "Invalid arguments for ask_data: question" in s.observation for s in run.steps)

    # An action that isn't one of the agent's tools fails schema validation twice -> error, no crash.
    unknown = run_agent(make_agent(retail, "automl"), "x", ScriptedProvider([{"action": "delete_files"}, {"action": "delete_files"}]))
    assert unknown.status == "error" and "language model" in unknown.note


def test_tool_errors_do_not_escape():
    def fail(**_):
        raise ValueError("no such column")

    agent = Agent("test", "Test agent.", [Tool("broken", "Always fails.", NoArguments, fail)])
    run = run_agent(agent, "go", ScriptedProvider([{"action": "broken"}, {"action": "final_answer", "answer": "It failed."}]))
    assert run.steps[0].observation == "broken failed: no such column" and run.status == "done"


def test_agents_need_a_language_model(retail):
    with pytest.raises(ValueError, match="need a language model"):
        start_agent_job(retail, "automl", "hello")
    with pytest.raises(ValueError, match="Unknown agent"):
        make_agent(retail, "hacker")


def test_agent_api(retail):
    from api.main import app, store as api_store

    api_store._items[retail.id] = retail
    client = TestClient(app)
    names = [a["name"] for a in client.get("/agents").json()]
    assert names == ["automl", "analyst"]

    use_provider(ScriptedProvider([
        {"action": "describe_dataset"},
        {"action": "final_answer", "answer": "The data has 800 rows."},
    ]))
    started = client.post(f"/datasets/{retail.id}/agents/analyst", json={"message": "how big is the data?"}).json()
    while (status := client.get(started["status_url"]).json())["status"] == "running":
        time.sleep(0.2)
    result = status["result"]
    assert result["type"] == "agent" and result["answer"] == "The data has 800 rows." and result["note"] is None
    assert result["steps"][0]["action"] == "describe_dataset"
