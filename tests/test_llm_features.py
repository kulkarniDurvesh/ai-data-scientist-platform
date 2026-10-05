"""
Phase 6b: language-model fallback for the Ask box, grounded narratives, evaluation sets.

Synthetic data only; the language model is a scripted stand-in.
"""

import time

import pytest
from fastapi.testclient import TestClient

from core.llm import ScriptedProvider, reset_provider, use_provider
from core.narrate import dataset_facts, narrate, unsupported_numbers
from service.session import DatasetStore
from tests.sample_data import churn_panel, retail_dataframe


@pytest.fixture(autouse=True)
def no_model():
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


# ----------------------------------------------------------------------
# Ask fallback
# ----------------------------------------------------------------------

def test_model_reads_what_rules_cannot_and_pandas_computes(retail, frame):
    use_provider(ScriptedProvider([{"intent": "SUM", "entity": "Sales", "filters": [{"column": "Region", "value": "south"}]}]))
    entry = retail.ask("how much did we make in the south")
    expected = round(frame.loc[frame["Region"] == "South", "Sales"].sum(), 2)
    assert entry["by_model"] and entry["error"] is None
    assert f"{expected}" in entry["answer"] or f"{expected:,.2f}" in entry["answer"]
    assert "Read with the language model as: sum of 'Sales' where Region = South" in entry["answer"]
    assert entry["plan"]["filters"] == [{"column": "Region", "operator": "=", "value": "South"}]


def test_model_values_and_columns_are_checked(retail):
    use_provider(ScriptedProvider([
        {"intent": "SUM", "entity": "Sales", "filters": [{"column": "Region", "value": "Atlantis"}]},
        {"intent": "AVERAGE", "entity": "Revenue"},
        {"answerable": False, "reason": "There is no product family column."},
    ]))
    unknown_value = retail.ask("how much did we make in atlantis")
    assert unknown_value["answer"] is None and "'Atlantis' is not a value of Region" in unknown_value["error"]

    invented = retail.ask("what is the typical takings")
    assert invented["answer"] is None and "'Revenue' is not a column" in invented["error"]

    unanswerable = retail.ask("which product family earns the most")
    assert unanswerable["error"] == "This data can't answer that: There is no product family column."


def test_values_named_in_the_question_are_always_filtered(retail, frame):
    # Found with the real model: it dropped the filter and counted everyone.
    use_provider(ScriptedProvider([{"intent": "COUNT", "entity": "customer_id"}]))
    entry = retail.ask("count shoppers in West")
    expected = frame.loc[frame["Region"] == "West", "customer_id"].nunique()
    assert entry["plan"]["filters"] == [{"column": "Region", "operator": "=", "value": "West"}]
    assert str(expected) in entry["answer"]


def test_rules_answer_without_calling_the_model(retail):
    provider = ScriptedProvider([])
    use_provider(provider)
    entry = retail.ask("total sales by region")
    assert entry["answer"] and not entry["by_model"] and not provider.calls


def test_model_failure_keeps_the_rule_message(retail):
    use_provider(ScriptedProvider(["not json", "still not json"]))
    entry = retail.ask("how much did we make in the south")
    assert entry["answer"] is None and "language model could not be used" in entry["error"]


def test_unknown_filter_value_is_refused_not_ignored(retail):
    # Regression (found by the evaluation set): the filter used to be dropped.
    entry = retail.ask("how many customers are in Atlantis region")
    assert entry["answer"] is None and "'Atlantis'" in entry["notice"]
    assert retail.ask("how many customers are in North region")["answer"].startswith("There are")
    assert retail.ask("bar chart of Sales by Region")["error"] is None


# ----------------------------------------------------------------------
# Narratives
# ----------------------------------------------------------------------

def test_dataset_summary_from_computed_facts(retail):
    summary = retail.narrative("dataset")
    assert summary.source == "template"
    assert "800 rows and 15 columns" in summary.text
    assert "Days To Ship" in summary.text  # the planted quality problem


def test_model_summary(retail):
    stores = DatasetStore().load(churn_panel(12, 120).to_csv(index=False).encode(), "stores.csv")
    with pytest.raises(ValueError):
        stores.narrative("model")
    job = stores.start_model_job("rank", "will_churn")
    while stores.model_job(job)["status"] == "running":
        time.sleep(0.3)
    text = stores.narrative("model").text
    assert "Will Churn" in text and "PR-AUC" in text and "Complaints" in text


def test_model_text_is_accepted_only_when_every_number_is_grounded(retail):
    facts = dataset_facts(retail.name, retail.df, retail.schema, retail.quality(), retail.insights(), retail.sheets)
    grounded = f"The table holds {facts.get('rows')} rows across {facts.get('columns')} columns."
    accepted = narrate(facts, ScriptedProvider([grounded]))
    assert accepted.source == "language model" and accepted.text == grounded

    invented = narrate(facts, ScriptedProvider(["The table holds 812 rows and sales grew 37.5% last year."]))
    assert invented.source == "template" and set(invented.unsupported) == {"812", "37.5"}
    assert "not in the computed results" in invented.note

    assert unsupported_numbers("Two tables and 3 issues.", facts) == []


def test_narrative_and_plan_in_the_api(retail):
    from api.main import app, store as api_store

    api_store._items[retail.id] = retail
    client = TestClient(app)
    summary = client.get(f"/datasets/{retail.id}/narrative").json()
    assert summary["source"] == "template" and summary["facts"]["rows"] == "800"
    assert client.get(f"/datasets/{retail.id}/narrative", params={"kind": "model"}).status_code == 404

    answer = client.post(f"/datasets/{retail.id}/ask", json={"question": "total sales by region"}).json()
    assert answer["plan"]["operation"] == "SUM" and answer["plan"]["group_by"] == "Region" and not answer["by_model"]


# ----------------------------------------------------------------------
# Evaluation sets
# ----------------------------------------------------------------------

def test_rules_pass_every_plain_case():
    from evals.runner import datasets, report, run_goals, run_questions

    bundles = datasets()
    results = run_goals(bundles, None) + run_questions(bundles, None)
    plain = [r for r in results if not r.needs_model]
    assert len(plain) >= 25
    assert all(r.passed for r in plain), report(results, "rules only")
    assert "| goals | plain |" in report(results, "rules only")


def test_scripted_model_on_the_synonym_questions():
    from evals.runner import datasets, run_questions

    bundles = datasets()
    provider = ScriptedProvider([
        {"intent": "SUM", "entity": "Sales", "filters": [{"column": "Region", "value": "South"}]},
        {"intent": "AVERAGE", "entity": "Sales", "filters": [{"column": "Region", "value": "North"}]},
        {"intent": "COUNT", "entity": "customer_id", "filters": [{"column": "Region", "value": "West"}]},
    ])
    results = [r for r in run_questions(bundles, provider) if r.needs_model]
    assert all(r.passed and r.source == "language model" for r in results), [r.mismatches for r in results]
