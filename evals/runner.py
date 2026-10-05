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


def report(results: list[CaseResult], mode: str) -> str:
    lines = [f"# Evaluation report ({mode})", ""]
    lines += ["| Set | Wording | Passed | Accuracy | Mean seconds |", "|---|---|---|---|---|"]
    for set_name in ("goals", "questions"):
        for needs_model, label in ((False, "plain"), (True, "needs a model")):
            group = [r for r in results if r.set == set_name and r.needs_model == needs_model]
            if not group:
                continue
            passed = sum(r.passed for r in group)
            seconds = sum(r.seconds for r in group) / len(group)
            lines.append(f"| {set_name} | {label} | {passed}/{len(group)} | {passed / len(group):.0%} | {seconds:.1f} |")
    failures = [r for r in results if not r.passed]
    if failures:
        lines += ["", "## Failures", ""]
        for r in failures:
            lines.append(f"- **{r.set} / {r.id}** ({r.source}): " + "; ".join(r.mismatches))
    return "\n".join(lines) + "\n"
