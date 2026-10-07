"""
Evaluation runner: score goal and question interpretation on fixed sets.

    python -m evals                    # rules only
    python -m evals --model            # rules + the configured language model
    python -m evals --out report.md    # also write the Markdown report

Cases run on synthetic datasets (the same generators as the tests), so
the sets need no private data. Each case passes when every expected
field matches; the report splits plain wording (rules must pass) from
wording that needs a language model.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from core.intent import interpret
from core.llm import Provider, use_provider
from service.session import DatasetBundle, DatasetStore

FOLDER = Path(__file__).resolve().parent


@dataclass
class CaseResult:
    set: str
    id: str
    needs_model: bool
    passed: bool
    mismatches: list[str] = field(default_factory=list)
    source: str = ""
    seconds: float = 0.0


def datasets() -> dict[str, DatasetBundle]:
    """The synthetic datasets the sets refer to."""

    from tests.sample_data import churn_panel, retail_dataframe
    from tests.test_forecast import _orders
    from tests.test_recommend import _workbook

    store = DatasetStore(capacity=8)
    csv = lambda frame: frame.to_csv(index=False).encode()  # noqa: E731
    return {
        "retail": store.load(csv(retail_dataframe(1500)), "retail.csv"),
        "stores": store.load(csv(churn_panel(rows_per_store=12, stores=120)), "stores.csv"),
        "sales": store.load(_workbook(), "sales.xlsx", "Contacts"),
        "orders": store.load(csv(_orders(months=36)), "orders.csv"),
    }


def load_set(name: str) -> list[dict[str, Any]]:
    return yaml.safe_load((FOLDER / f"{name}.yaml").read_text(encoding="utf-8")) or []


def _matches(expected: Any, actual: Any) -> bool:
    if isinstance(expected, list) and not isinstance(actual, list):
        return actual in expected
    if isinstance(expected, list):
        return all(item in (actual or []) for item in expected)
    return expected == actual


def run_goals(bundles: dict[str, DatasetBundle], provider: Provider | None) -> list[CaseResult]:
    results = []
    for case in load_set("goals"):
        bundle = bundles[case["dataset"]]
        start = time.time()
        plan = interpret(case["text"], bundle.goal_view(), provider)
        spec = plan.spec.to_dict()
        mismatches = [
            f"{key}: expected {value!r}, got {spec.get(key)!r}"
            for key, value in case["expect"].items()
            if not _matches(value, spec.get(key))
        ]
        if case["expect"].get("task") is not None and not plan.ready:
            mismatches.append("plan not ready: " + "; ".join(q.text for q in plan.questions))
        results.append(CaseResult("goals", case["id"], bool(case.get("needs_model")), not mismatches,
                                  mismatches, plan.source, time.time() - start))
    return results


def run_questions(bundles: dict[str, DatasetBundle], provider: Provider | None) -> list[CaseResult]:
    use_provider(provider)
    results = []
    for case in load_set("questions"):
        bundle = bundles[case["dataset"]]
        start = time.time()
        entry = bundle.ask(case["question"])
        plan = entry.get("plan") or {}
        mismatches: list[str] = []

        if case.get("refuse"):
            if entry.get("answer"):
                mismatches.append(f"expected a refusal, got: {entry['answer']}")
        elif not entry.get("answer"):
            mismatches.append(f"no answer: {entry.get('error') or entry.get('notice')}")
        else:
            for key, value in case["expect"].items():
                actual = plan.get(key)
                if key == "filters":
                    actual = [[f["column"], f["operator"], str(f["value"])] for f in actual or []]
                    value = [[c, o, str(v)] for c, o, v in value]
                    ok = all(item in actual for item in value)
                else:
                    ok = value == actual
                if not ok:
                    mismatches.append(f"{key}: expected {value!r}, got {actual!r}")

        source = "language model" if entry.get("by_model") else "rules"
        results.append(CaseResult("questions", case["id"], bool(case.get("needs_model")), not mismatches,
                                  mismatches, source, time.time() - start))
    use_provider(None)
    return results


def run_documents(knowledge=None, use_model: bool = False) -> list[CaseResult]:
    """Retrieval: is the answering passage in the top 3? Refusals: is the answer 'not covered'?"""

    from service.knowledge import KnowledgeBase

    knowledge = knowledge or KnowledgeBase(cache=None)
    index = knowledge.index()
    results = []
    for case in load_set("documents"):
        start = time.time()
        mismatches: list[str] = []
        rank = None
        if case.get("not_covered"):
            answer = knowledge.ask(case["question"], use_model)
            if answer.covered:
                mismatches.append(f"expected 'not covered', got: {answer.text[:80]}")
        else:
            hits = index.search(case["question"], k=10)
            expect = case["expect"]
            for position, hit in enumerate(hits, 1):
                if hit.chunk.source.endswith(expect["source"]) and expect["text"].lower() in hit.chunk.text.lower():
                    rank = position
                    break
            if rank is None or rank > 3:
                mismatches.append(f"answering passage at rank {rank or '>10'}; top: {hits[0].chunk.section if hits else '-'}")
        result = CaseResult("documents", case["id"], bool(case.get("paraphrase")), not mismatches,
                            mismatches, index.embedder_name, time.time() - start)
        result.rank = rank
        results.append(result)
    return results


def run_agents(bundles: dict[str, DatasetBundle], provider: Provider) -> list[CaseResult]:
    """Agents with a real model: finished, used the expected tools, answer has the expected words."""

    from core.agents import run_agent
    from service.agents import make_agent

    results = []
    for case in load_set("agents"):
        run = run_agent(make_agent(bundles[case["dataset"]], case["agent"]), case["message"], provider)
        used = [s.action for s in run.steps]
        mismatches = []
        if run.status not in ("done", "needs_input"):
            mismatches.append(f"status {run.status}: {run.note or ''}")
        mismatches += [f"tool {t} not used (used {used})" for t in case.get("expect_tools", []) if t not in used]
        mismatches += [f"answer lacks '{w}'" for w in case.get("expect_answer", []) if w.lower() not in run.answer.lower()]
        if run.note and run.status == "done":
            mismatches.append(run.note)
        results.append(CaseResult("agents", case["id"], True, not mismatches, mismatches, provider.describe(), run.seconds))
    return results


def retrieval_metrics(results: list[CaseResult]) -> dict[str, float]:
    ranked = [r for r in results if r.set == "documents" and hasattr(r, "rank") and r.id and not r.id.startswith("off-topic")]
    ranks = [getattr(r, "rank", None) for r in ranked]
    n = len(ranks) or 1
    return {
        "hit@1": sum(1 for x in ranks if x == 1) / n,
        "hit@3": sum(1 for x in ranks if x and x <= 3) / n,
        "mrr": sum(1 / x for x in ranks if x) / n,
    }


def report(results: list[CaseResult], mode: str) -> str:
    lines = [f"# Evaluation report ({mode})", ""]
    lines += ["| Set | Wording | Passed | Accuracy | Mean seconds |", "|---|---|---|---|---|"]
    for set_name in ("goals", "questions", "documents", "agents"):
        labels = {"documents": ((False, "plain"), (True, "paraphrase")), "agents": ((True, "with a model"),)}.get(
            set_name, ((False, "plain"), (True, "needs a model")))
        for needs_model, label in labels:
            group = [r for r in results if r.set == set_name and r.needs_model == needs_model]
            if not group:
                continue
            passed = sum(r.passed for r in group)
            seconds = sum(r.seconds for r in group) / len(group)
            lines.append(f"| {set_name} | {label} | {passed}/{len(group)} | {passed / len(group):.0%} | {seconds:.1f} |")
    documents = [r for r in results if r.set == "documents"]
    if documents:
        for label, group in (("plain", [r for r in documents if not r.needs_model]), ("paraphrase", [r for r in documents if r.needs_model])):
            metrics = retrieval_metrics(group)
            lines.append("")
            lines.append(f"Retrieval ({label}, embedder {documents[0].source}): hit@1 {metrics['hit@1']:.0%}, "
                         f"hit@3 {metrics['hit@3']:.0%}, MRR {metrics['mrr']:.2f}")
    failures = [r for r in results if not r.passed]
    if failures:
        lines += ["", "## Failures", ""]
        for r in failures:
            lines.append(f"- **{r.set} / {r.id}** ({r.source}): " + "; ".join(r.mismatches))
    return "\n".join(lines) + "\n"
