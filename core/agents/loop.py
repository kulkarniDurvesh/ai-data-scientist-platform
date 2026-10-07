"""
A hand-written tool-calling agent loop.

    user message
      -> model returns one step as JSON: {thought, action, arguments, answer}
      -> action is a tool: validate arguments, run it, append the observation
      -> action is "final_answer": check that its numbers come from
         observations, then stop
      -> stop after MAX_STEPS with what was found

The model chooses tools; tools compute. Every step is kept in a trace
(thought, tool, arguments, observation, time) so a run can be inspected.
The JSON-step protocol works with any chat model (Ollama, Azure OpenAI,
a scripted stand-in), with or without native function calling.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field, create_model

from core.llm import LLMError, Provider, structured

from .tools import Tool

MAX_STEPS = 6
FINAL = "final_answer"
NUMBER = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?")
FREE_NUMBERS = {str(n) for n in range(0, 11)}


@dataclass
class Step:
    thought: str
    action: str
    arguments: dict[str, Any]
    observation: str = ""
    ok: bool = True
    seconds: float = 0.0

    def to_dict(self) -> dict:
        return {"thought": self.thought, "action": self.action, "arguments": self.arguments,
                "observation": self.observation, "ok": self.ok, "seconds": round(self.seconds, 1)}


@dataclass
class AgentRun:
    agent: str
    message: str
    answer: str = ""
    status: str = "done"                 # done | needs_input | step_limit | error
    steps: list[Step] = field(default_factory=list)
    note: str | None = None
    seconds: float = 0.0

    def to_dict(self) -> dict:
        return {"agent": self.agent, "message": self.message, "answer": self.answer, "status": self.status,
                "steps": [s.to_dict() for s in self.steps], "note": self.note, "seconds": round(self.seconds, 1)}


@dataclass
class Agent:
    name: str
    instructions: str
    tools: list[Tool]

    def tool(self, name: str) -> Tool | None:
        return next((t for t in self.tools if t.name == name), None)


def step_model(agent: Agent) -> type[BaseModel]:
    actions = tuple(t.name for t in agent.tools) + (FINAL,)
    return create_model(
        "AgentStep",
        thought=(str, Field("", description="One short sentence: why this action")),
        action=(Literal[actions], Field(..., description="A tool name, or final_answer")),
        arguments=(dict[str, Any], Field(default_factory=dict, description="Arguments for the tool")),
        answer=(str | None, Field(None, description="The final answer, when action is final_answer")),
    )


def system_prompt(agent: Agent) -> str:
    tools = "\n".join(t.describe() for t in agent.tools)
    return f"""{agent.instructions}

Tools:
{tools}

Reply with one JSON step at a time: {{"thought": ..., "action": <tool name or "{FINAL}">, "arguments": {{...}}, "answer": ...}}.
Rules:
- Use tools to get facts. Never compute or invent numbers yourself; copy them from tool results.
- After each tool call you will see its result as an observation.
- Finish with action "{FINAL}" and a short answer that cites what the tools found.
- If a tool says something is missing or unclear, ask the user with the ask_user tool if you have it."""


def _normal(number: str) -> str:
    return number.rstrip("0").rstrip(".") if "." in number else number


def unsupported_numbers(answer: str, steps: list[Step], message: str) -> list[str]:
    """
    Numbers in the answer that no tool returned. Rounding an observed
    number (43816.27 -> 43,816 or 43816.3) is reporting it, not inventing it.
    """

    seen = " ".join([message] + [s.observation for s in steps]).replace(",", "")
    allowed = set()
    for number in NUMBER.findall(seen):
        allowed.add(_normal(number))
        if "." in number:
            value = float(number)
            allowed |= {str(round(value)), _normal(f"{value:.1f}"), _normal(f"{value:.2f}")}
    found = []
    for number in NUMBER.findall(answer):
        value = _normal(number.replace(",", ""))
        if value not in FREE_NUMBERS and value not in allowed and number not in found:
            found.append(number)
    return found


def run_agent(
    agent: Agent,
    message: str,
    provider: Provider,
    history: list[dict[str, str]] | None = None,
    progress=None,
    max_steps: int = MAX_STEPS,
) -> AgentRun:
    """Run one user turn; `history` holds earlier turns of the conversation."""

    start = time.time()
    run = AgentRun(agent.name, message)
    schema = step_model(agent)
    messages = [{"role": "system", "content": system_prompt(agent)}]
    messages += list(history or [])
    messages.append({"role": "user", "content": message})
    report = progress or (lambda text: None)

    for _ in range(max_steps):
        step_start = time.time()
        try:
            step = structured(provider, messages[0]["content"], _transcript(messages[1:]), schema)
        except LLMError as error:
            run.status, run.note = "error", f"The language model could not be used: {error}"
            break

        if step.action == FINAL:
            answer = (step.answer or step.thought or "").strip()
            unsupported = unsupported_numbers(answer, run.steps, message)
            if unsupported:
                run.note = ("The answer contained numbers no tool returned (" + ", ".join(unsupported)
                            + "); see the tool results below.")
            run.answer = answer
            run.steps.append(Step(step.thought, FINAL, {}, seconds=time.time() - step_start))
            break

        tool = agent.tool(step.action)
        ok, observation = tool.call(step.arguments)
        record = Step(step.thought, tool.name, step.arguments, observation, ok, time.time() - step_start)
        run.steps.append(record)
        report(f"{tool.name}: {observation[:120]}")
        messages.append({"role": "assistant", "content": f"Action: {tool.name} {step.arguments}"})
        messages.append({"role": "user", "content": f"Observation: {observation}"})

        if tool.stops and ok:
            run.status, run.answer = "needs_input", observation
            break
    else:
        run.status = "step_limit"
        run.answer = "I stopped after the step limit. What the tools found is listed below."

    run.seconds = time.time() - start
    return run


def _transcript(messages: list[dict[str, str]]) -> str:
    """Earlier turns and this turn's steps as one user message (works with any chat model)."""

    lines = []
    for item in messages:
        prefix = {"user": "User", "assistant": "You"}.get(item["role"], item["role"])
        content = item["content"]
        if content.startswith("Observation:"):
            prefix = "Tool"
        lines.append(f"{prefix}: {content}")
    lines.append("Next step (JSON):")
    return "\n".join(lines)


def history_from(run: AgentRun) -> list[dict[str, str]]:
    """The turn as conversation history for the next turn (question, tools used, answer)."""

    used = ", ".join(s.action for s in run.steps if s.action != FINAL) or "no tools"
    return [
        {"role": "user", "content": run.message},
        {"role": "assistant", "content": f"(used {used}) {run.answer}"},
    ]
