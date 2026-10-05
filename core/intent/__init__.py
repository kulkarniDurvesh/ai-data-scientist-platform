"""Free-text goal box: rules + language model -> validated plan; suggestions from the data (Phase 6)."""

from .complete import complete, describe, match_column
from .engine import interpret, suggestions, task_labels
from .rules import read_goal
from .spec import MODEL_TASKS, TASKS, GoalPlan, IntentSpec, Question
from .view import DataView

__all__ = [
    "MODEL_TASKS",
    "TASKS",
    "DataView",
    "GoalPlan",
    "IntentSpec",
    "Question",
    "complete",
    "describe",
    "interpret",
    "match_column",
    "read_goal",
    "suggestions",
    "task_labels",
]
