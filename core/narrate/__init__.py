"""Narratives grounded in computed results: fact sheets, templates, checked language-model text (Phase 6b)."""

from .facts import Fact, FactSheet, dataset_facts, model_facts
from .write import Narrative, narrate, template_text, unsupported_numbers

__all__ = [
    "Fact",
    "FactSheet",
    "Narrative",
    "dataset_facts",
    "model_facts",
    "narrate",
    "template_text",
    "unsupported_numbers",
]
