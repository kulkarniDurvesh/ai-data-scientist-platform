"""
Narratives from fact sheets: a template always, a language model on request.

A model's text is accepted only if every number in it appears in the
fact sheet (grounding check); otherwise the template text is shown with
a note naming the numbers that weren't in the computed results.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from core.llm import LLMError, Provider

from .facts import FactSheet

NUMBER = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?")
# Small whole numbers ("top 3", "two tables") carry no claim worth checking.
FREE_NUMBERS = {str(n) for n in range(0, 11)}


@dataclass
class Narrative:
    text: str
    source: str                         # template | language model
    note: str | None = None
    unsupported: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"text": self.text, "source": self.source, "note": self.note, "unsupported": list(self.unsupported)}


def template_text(sheet: FactSheet) -> str:
    get = sheet.get
    if sheet.kind == "dataset":
        parts = [f"The data has {get('rows')} rows and {get('columns')} columns"
                 + (f" across {get('tables in the file')} tables" if get("tables in the file") else "")
                 + f" ({get('column roles')})."]
        if get("time span"):
            parts.append(f"It runs from {get('time span')} on {get('time axis')}.")
        serious = get("findings needing attention")
        verb = "needs" if serious == "1" else "need"
        parts.append(f"{get('missing cells')} of cells are empty. Of {get('quality findings')} quality findings, "
                     f"{serious} {verb} attention.")
        issues = [get(f"quality issue {i}") for i in (1, 2) if get(f"quality issue {i}")]
        if issues:
            parts.append("Most important: " + " ".join(_sentence(i) for i in issues))
        insights = [get(f"insight {i}") for i in (1, 2, 3) if get(f"insight {i}")]
        if insights:
            parts.append("Notable patterns: " + " ".join(_sentence(i) for i in insights))
        return " ".join(parts)

    parts = [f"Goal: {get('goal')}. {get('chosen model')} was chosen and scored once on a held-out test period."]
    if get("PR-AUC"):
        parts.append(f"On the test period {get('top of the list')}, {get('lift at the top')}; "
                     f"PR-AUC {get('PR-AUC')}, ROC-AUC {get('ROC-AUC')}.")
    elif get("MAE"):
        parts.append(f"Test MAE {get('MAE')}" + (f", {get('error reduction')}" if get("error reduction") else "") + ".")
    if get("strongest features"):
        parts.append(f"The strongest signals are {get('strongest features')}.")
    warnings = [get(f"warning {i}") for i in (1, 2) if get(f"warning {i}")]
    parts += [_sentence(w) for w in warnings]
    return " ".join(parts)


def _sentence(text: str) -> str:
    text = text.strip()
    return text if text.endswith((".", "!", "?")) else text + "."


def _normal(number: str) -> str:
    value = number.replace(",", "")
    if "." in value:
        value = value.rstrip("0").rstrip(".")
    return value


def unsupported_numbers(text: str, sheet: FactSheet) -> list[str]:
    allowed = {_normal(n) for fact in sheet.facts for n in NUMBER.findall(fact.key + " " + fact.text)}
    allowed |= {_normal(n) for n in NUMBER.findall(sheet.title)}
    found = []
    for number in NUMBER.findall(text):
        value = _normal(number)
        if value in FREE_NUMBERS or value in allowed or number in found:
            continue
        found.append(number)
    return found


SYSTEM = """You write a short summary for a business reader from a list of computed facts.
Rules:
- Use only the facts given. Do not add, round, recompute or estimate any number; copy numbers exactly.
- 3 to 5 plain sentences, no headings, no lists, no recommendations beyond the facts.
- Mention warnings or quality problems if they are listed."""


def narrate(sheet: FactSheet, provider: Provider | None = None) -> Narrative:
    template = template_text(sheet)
    if provider is None:
        return Narrative(template, "template")

    try:
        text = provider.chat(
            [{"role": "system", "content": SYSTEM},
             {"role": "user", "content": f"Facts about {sheet.title}:\n{sheet.lines()}\n\nSummary:"}],
        ).strip()
    except LLMError as error:
        return Narrative(template, "template", note=f"The language model could not be used ({error}).")

    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    unsupported = unsupported_numbers(text, sheet)
    if unsupported or not text:
        return Narrative(
            template, "template",
            note="The language model's summary used numbers that are not in the computed results ("
                 + ", ".join(unsupported) + "), so the computed summary is shown." if unsupported
                 else "The language model returned no text.",
            unsupported=unsupported,
        )
    return Narrative(text, "language model")
