"""
The same agents on Microsoft Agent Framework (Phase 8b).

The hand-written loop (core/agents) shows every part of an agent; this
module runs the same instructions and tools with Agent Framework's Agent
and native function calling, through its OpenAI-compatible client:

    AIDS_LLM_PROVIDER=ollama  -> Ollama's OpenAI-compatible endpoint (<host>/v1)
    AIDS_LLM_PROVIDER=azure   -> Azure OpenAI (endpoint, deployment, key, API version)

Tool calls are recorded as steps, so a framework run returns the same
AgentRun (answer, steps, number check) as the hand-written loop.
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any

from core.agents import AgentRun, Step, unsupported_numbers
from core.llm.provider import DEFAULT_AZURE_API_VERSION, DEFAULT_OLLAMA_HOST, DEFAULT_OLLAMA_MODEL

from .agents import AGENTS, build_tools
from .session import DatasetBundle

RULES = """
Use the tools to get facts; never compute or invent numbers yourself - copy them from tool results.
If a plan has questions, ask the user with ask_user and stop."""


def framework_client():
    """An Agent Framework chat client for the configured provider."""

    from agent_framework.openai import OpenAIChatCompletionClient

    choice = os.environ.get("AIDS_LLM_PROVIDER", "none").strip().lower()
    if choice == "azure" or (choice == "auto" and os.environ.get("AIDS_AZURE_OPENAI_ENDPOINT")):
        return OpenAIChatCompletionClient(
            model=os.environ.get("AIDS_AZURE_OPENAI_DEPLOYMENT"),
            api_key=os.environ.get("AIDS_AZURE_OPENAI_KEY"),
            azure_endpoint=os.environ.get("AIDS_AZURE_OPENAI_ENDPOINT"),
            api_version=os.environ.get("AIDS_AZURE_OPENAI_API_VERSION", DEFAULT_AZURE_API_VERSION),
        )
    if choice in ("ollama", "auto"):
        host = (os.environ.get("AIDS_OLLAMA_HOST") or DEFAULT_OLLAMA_HOST).rstrip("/")
        return OpenAIChatCompletionClient(
            model=os.environ.get("AIDS_LLM_MODEL") or DEFAULT_OLLAMA_MODEL,
            api_key="ollama",
            base_url=f"{host}/v1",
        )
    raise ValueError("Agents need a language model: set AIDS_LLM_PROVIDER=ollama or azure.")


def run_framework_agent(
    bundle: DatasetBundle,
    name: str,
    message: str,
    client=None,
    history: list[dict[str, str]] | None = None,
) -> AgentRun:
    from agent_framework import Agent, tool

    if name not in AGENTS:
        raise ValueError(f"Unknown agent '{name}'. Agents: {', '.join(AGENTS)}.")
    label, instructions, tool_names = AGENTS[name]
    tools = build_tools(bundle)
    run = AgentRun(f"{label} (Agent Framework)", message)
    start = time.time()

    def wrap(platform_tool):
        def invoke(**arguments: Any) -> str:
            began = time.time()
            ok, observation = platform_tool.call(arguments)
            run.steps.append(Step("", platform_tool.name, arguments, observation, ok, time.time() - began))
            return observation

        return tool(invoke, name=platform_tool.name, description=platform_tool.description, schema=platform_tool.arguments)

    agent = Agent(
        client or framework_client(),
        instructions=instructions + RULES,
        name=name,
        tools=[wrap(tools[t]) for t in tool_names],
    )
    earlier = "\n".join(f"{'User' if m['role'] == 'user' else 'You'}: {m['content']}" for m in history or [])
    prompt = f"Earlier in this conversation:\n{earlier}\n\nUser: {message}" if earlier else message

    try:
        response = asyncio.run(agent.run(prompt))
        run.answer = (response.text or "").strip()
    except Exception as error:  # noqa: BLE001 - shown to the user
        run.status, run.note = "error", f"The agent failed: {error}"
        run.seconds = time.time() - start
        return run

    if run.steps and run.steps[-1].action == "ask_user" and run.steps[-1].ok:
        run.status, run.answer = "needs_input", run.steps[-1].observation
    unsupported = unsupported_numbers(run.answer, run.steps, message)
    if unsupported and run.status == "done":
        run.note = "The answer contained numbers no tool returned (" + ", ".join(unsupported) + "); see the tool results below."
    run.seconds = time.time() - start
    return run
