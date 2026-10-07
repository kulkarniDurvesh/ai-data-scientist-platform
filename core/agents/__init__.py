"""Agents: typed tools and a hand-written tool-calling loop with a trace (Phase 8)."""

from .loop import FINAL, Agent, AgentRun, Step, history_from, run_agent, step_model, system_prompt, unsupported_numbers
from .tools import NoArguments, Tool

__all__ = [
    "FINAL",
    "Agent",
    "AgentRun",
    "NoArguments",
    "Step",
    "Tool",
    "history_from",
    "run_agent",
    "step_model",
    "system_prompt",
    "unsupported_numbers",
]
