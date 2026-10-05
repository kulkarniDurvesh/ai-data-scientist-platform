"""Goal-driven model building (Phase 3a: classification, regression, ranking)."""

from .builder import ModelResult, build_model
from .goal import GOAL_TYPES, GoalSpec, propose_goal, target_options
from .registry import list_models, load_model, save_model

__all__ = [
    "GOAL_TYPES",
    "GoalSpec",
    "ModelResult",
    "build_model",
    "list_models",
    "load_model",
    "propose_goal",
    "save_model",
    "target_options",
]
