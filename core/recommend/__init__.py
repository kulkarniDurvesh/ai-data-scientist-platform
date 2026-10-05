"""Recommendation: next-best-contact ranking, daily plans and backtests (Phase 3b)."""

from .engine import RecommendResult, build_recommender
from .planner import PlanSettings
from .roles import InteractionRoles, complete_roles, default_roles, group_options, role_options, suggest_success

__all__ = [
    "InteractionRoles",
    "PlanSettings",
    "RecommendResult",
    "build_recommender",
    "complete_roles",
    "default_roles",
    "group_options",
    "role_options",
    "suggest_success",
]
