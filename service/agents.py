"""
The platform's agents (Phase 8): tools bound to a dataset, two agents,
conversations and background runs.

    automl    goal -> interpret -> clarifying questions -> run the pipeline
              -> explain the result
    analyst   questions about the data and the documents

Tools call the same session methods as the dashboard and the API, so an
agent can do nothing a user couldn't, and every number comes from them.
Tools are deterministic: they never call the language model themselves
(the agent already is the model), which keeps runs predictable and avoids
nested model calls.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field

from core.agents import Agent, AgentRun, NoArguments, Tool, history_from, run_agent
from core.llm import get_provider
from core.schema_inference import format_number

from .knowledge import knowledge
from .session import DatasetBundle

JOB_TIMEOUT = 600
TABLE_ROWS = 8


class Question(BaseModel):
    question: str = Field(..., description="A question in plain words")


class GoalText(BaseModel):
    text: str = Field(..., description="The user's goal in plain words")


class GoalSpecArgs(BaseModel):
    spec: dict[str, Any] = Field(..., description="The 'spec' returned by interpret_goal, with any answers applied")


class SearchArgs(BaseModel):
    query: str = Field(..., description="What to look for")


class AskUserArgs(BaseModel):
    question: str = Field(..., description="One clear question for the user, with the options if there are any")


def _table(frame: pd.DataFrame | None, rows: int = TABLE_ROWS) -> str | None:
    if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty:
        return None
    head = frame.head(rows).copy()
    numeric = head.select_dtypes("number").columns
    head[numeric] = head[numeric].round(3)
    return head.to_string(index=False)


def build_tools(bundle: DatasetBundle) -> dict[str, Tool]:
    def describe_dataset() -> dict:
        facts = bundle.facts("dataset").to_dict()["facts"]
        columns = [f"{c} ({bundle.schema.role_of(c)})" for c in bundle.df.columns][:60]
        return {"facts": facts, "sheets": [s for s in bundle.sheets], "loaded table": bundle.sheet, "columns": columns}

    def ask_data(question: str) -> dict:
        entry = bundle.ask(question, use_model=False)
        if entry.get("error") or entry.get("notice"):
            return {"answer": None, "problem": entry.get("error") or entry.get("notice")}
        return {"answer": entry["answer"], "table": _table(entry.get("table"))}

    def ask_documents(question: str) -> dict:
        answer = knowledge.ask(question)
        return {
            "covered": answer.covered,
            "answer": answer.text,
            "sources": [f"[{s.number}] {s.hit.chunk.section} ({s.hit.chunk.source})" for s in answer.sources],
        }

    def search_documents(query: str) -> list[dict]:
        return [{"section": h.chunk.section, "source": h.chunk.source, "text": h.chunk.text[:500]}
                for h in knowledge.search(query, 3)]

    def suggest_goals() -> list[dict]:
        return [{"title": p.title, "summary": p.summary, "spec": _compact(p.spec.to_dict())} for p in bundle.goal_suggestions()]

    def interpret_goal(text: str) -> dict:
        plan = bundle.interpret_goal(text, use_model=False)
        return {
            "ready": plan.ready,
            "summary": plan.summary,
            "spec": _compact(plan.spec.to_dict()),
            "choices made": plan.assumptions,
            "questions": [{"field": q.field, "question": q.text, "options": q.options[:10]} for q in plan.questions],
            "warnings": plan.warnings,
        }

    def run_goal(spec: dict) -> dict:
        started = bundle.run_goal(spec)
        if "entry" in started:
            entry = started["entry"]
            return {"answer": entry.get("answer") or entry.get("error") or entry.get("notice")}
        job_id = started["job_id"]
        deadline = time.time() + JOB_TIMEOUT
        while time.time() < deadline:
            job = bundle.model_job(job_id)
            if job["status"] != "running":
                break
            time.sleep(0.5)
        if job["status"] == "error":
            return {"error": job["error"]}
        if job["status"] == "running":
            return {"error": "The pipeline is still running; check the results tab later."}
        return {"tab": started["tab"], **_result_summary(job["result"])}

    def summarize_model() -> dict:
        return bundle.narrative("model").to_dict() | {"facts": bundle.facts("model").to_dict()["facts"]}

    def ask_user(question: str) -> str:
        return question

    tools = [
        Tool("describe_dataset", "Rows, columns with their roles, tables, time span and data quality of the loaded data.", NoArguments, describe_dataset),
        Tool("ask_data", "Answer a question by computing on the data (counts, totals, averages, lists, top N, filters, by-group).", Question, ask_data),
        Tool("ask_documents", "Answer from the company documents and platform docs, with numbered sources.", Question, ask_documents),
        Tool("search_documents", "Find the document passages about a topic.", SearchArgs, search_documents),
        Tool("suggest_goals", "Goals this dataset supports (prediction, ranking, plans, forecasts, segments, explanations), each ready to run.", NoArguments, suggest_goals),
        Tool("interpret_goal", "Turn a goal in plain words into a checked plan: summary, spec, choices made and questions that need an answer.", GoalText, interpret_goal),
        Tool("run_goal", "Run a ready plan's spec with the platform's pipelines and return the result summary.", GoalSpecArgs, run_goal),
        Tool("summarize_model", "Plain-language summary and facts of the last trained model.", NoArguments, summarize_model),
        Tool("ask_user", "Ask the user one question and wait for the reply (ends this turn).", AskUserArgs, ask_user, stops=True),
    ]
    return {tool.name: tool for tool in tools}


def _compact(spec: dict) -> dict:
    return {key: value for key, value in spec.items() if value not in (None, [])}


def _result_summary(result: Any) -> dict:
    """What an agent needs from a pipeline result: its summary and a small table."""

    steps = getattr(result, "steps", None)
    summary = " ".join(s.text for s in steps) if steps else getattr(result, "summary", "")
    out: dict[str, Any] = {"summary": summary}
    for name in ("user_table", "forecasts", "importance", "attention"):
        table = getattr(result, name, None)
        if isinstance(table, pd.DataFrame) and not table.empty:
            out[name.replace("_", " ")] = _table(table)
            break
    clustering = getattr(result, "clustering", None)
    if clustering is not None:
        out["segments"] = _table(clustering.profiles)
    metrics = getattr(result, "test_metrics", None)
    if metrics:
        out["test metrics"] = {k: format_number(v) if isinstance(v, float) else v for k, v in metrics.items()}
    return out


AGENTS = {
    "automl": (
        "AutoML builder",
        "You are an AutoML assistant for a tabular dataset. You turn the user's goal into a model, plan, forecast, "
        "segmentation or explanation using the platform's pipelines.\n"
        "Steps: (1) if the goal is vague, call suggest_goals or describe_dataset; (2) call interpret_goal with the "
        "user's words; (3) if it has questions, ask_user one of them with its options; when the user answers, call "
        "interpret_goal again with the goal plus the answer; (4) when ready, call run_goal with the spec; (5) give a "
        "short final answer: what ran, the key result and where to see details.",
        ["describe_dataset", "suggest_goals", "interpret_goal", "run_goal", "summarize_model", "ask_user"],
    ),
    "analyst": (
        "Analyst",
        "You are a data analyst. Answer the user's question with facts from the data (ask_data) and from the company "
        "documents (ask_documents), combining both when useful. Use describe_dataset to learn the columns first if "
        "you are unsure what they are called. For 'why did X change' questions, interpret_goal and run_goal explain "
        "the change. Keep answers short and say which tool each fact came from.",
        ["describe_dataset", "ask_data", "ask_documents", "search_documents", "interpret_goal", "run_goal", "ask_user"],
    ),
}


def make_agent(bundle: DatasetBundle, name: str) -> Agent:
    if name not in AGENTS:
        raise ValueError(f"Unknown agent '{name}'. Agents: {', '.join(AGENTS)}.")
    label, instructions, tool_names = AGENTS[name]
    tools = build_tools(bundle)
    return Agent(label, instructions, [tools[t] for t in tool_names])


# Conversations: (dataset id, conversation id) -> history and runs
_conversations: dict[tuple[str, str], dict] = {}


def conversation(bundle: DatasetBundle, conversation_id: str | None) -> tuple[str, dict]:
    conversation_id = conversation_id or uuid.uuid4().hex[:8]
    key = (bundle.id, conversation_id)
    if key not in _conversations:
        _conversations[key] = {"history": [], "runs": []}
    return conversation_id, _conversations[key]


def start_agent_job(bundle: DatasetBundle, agent_name: str, message: str, conversation_id: str | None = None, provider=None) -> tuple[str, str]:
    """Run one turn in the background; returns (job id, conversation id)."""

    provider = provider or get_provider()
    if provider is None:
        raise ValueError("Agents need a language model: set AIDS_LLM_PROVIDER=ollama or azure.")
    agent = make_agent(bundle, agent_name)
    conversation_id, state = conversation(bundle, conversation_id)

    def work(progress):
        progress(f"{agent.name} is working on: {message}")
        return run_agent(agent, message, provider, state["history"], progress)

    def done(run: AgentRun):
        state["history"].extend(history_from(run))
        state["runs"].append(run)

    return bundle._run_job(work, done), conversation_id


def conversation_runs(bundle: DatasetBundle, conversation_id: str | None) -> list[AgentRun]:
    if not conversation_id:
        return []
    return list(_conversations.get((bundle.id, conversation_id), {}).get("runs", []))
