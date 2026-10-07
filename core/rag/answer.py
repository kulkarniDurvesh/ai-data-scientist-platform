"""
Answers from retrieved passages, always with citations.

    extractive (default, instant)  the passages' sentences that best match
                                   the question, each followed by [n]
    language model (optional)      a written answer from the numbered
                                   passages, then checked: every sentence
                                   needs a citation to a retrieved passage,
                                   and every number must appear in the
                                   passages it cites, else the extractive
                                   answer is shown with a note

A question the documents don't cover gets "not covered", not a guess.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from core.llm import LLMError, Provider

from .documents import words
from .search import Hit

MIN_COVERAGE = 0.6
# Cosine similarity above which a passage counts as on topic even without
# shared words (paraphrases). LSA only knows the documents' own words, so
# its similarity never overrides missing words.
MIN_SIMILARITY = {"lsa": 2.0}
DEFAULT_MIN_SIMILARITY = 0.72
SOURCES_FOR_ANSWER = 4
SENTENCES = 3
NUMBER = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?")
NOT_COVERED = "The documents don't cover this question."
# Words that shape a question rather than name its subject ("how long does it take").
QUESTION_WORDS = set(words(
    "long much many often take get need happen quick quickly fast soon ok okay tell know find give "
    "allowed allow possible way done make let"
))


@dataclass
class Source:
    number: int
    hit: Hit

    def to_dict(self) -> dict:
        return {"number": self.number, **self.hit.to_dict()}


@dataclass
class DocumentAnswer:
    question: str
    text: str
    covered: bool
    method: str                          # extractive | language model
    sources: list[Source] = field(default_factory=list)
    note: str | None = None
    grounded: float | None = None        # share of sentences with a valid, supported citation
    problems: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "question": self.question, "text": self.text, "covered": self.covered, "method": self.method,
            "sources": [s.to_dict() for s in self.sources], "note": self.note,
            "grounded": self.grounded, "problems": list(self.problems),
        }


def coverage(question: str, hits: list[Hit], idf: dict[str, float] | None = None) -> float:
    """
    Share of the question's words found in the top passages, weighted by
    rarity: matching "own" and "use" says little, missing "kilometre" a lot.
    Words that appear in no document get the highest weight.
    """

    terms = set(words(question)) - QUESTION_WORDS
    if not terms:
        return 0.0
    idf = idf or {}
    unseen = max(idf.values(), default=1.0)
    weight = {t: idf.get(t, unseen) for t in terms}
    found = {t for hit in hits[:3] for t in words(hit.chunk.searchable()) if t in terms}
    return sum(weight[t] for t in found) / sum(weight.values())


def is_covered(question: str, hits: list[Hit], embedder_name: str, idf: dict[str, float] | None = None) -> bool:
    if not hits:
        return False
    threshold = MIN_SIMILARITY.get(embedder_name, DEFAULT_MIN_SIMILARITY)
    best_similarity = max(h.similarity for h in hits[:3])
    return coverage(question, hits, idf) >= MIN_COVERAGE or best_similarity >= threshold


def _units(text: str) -> list[str]:
    """Sentences, list items and table rows: the pieces an answer can quote."""

    units = []
    for line in text.split("\n"):
        line = line.strip()
        if not line or set(line) <= set("|-: "):
            continue
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            units.append(" — ".join(c for c in cells if c))
            continue
        line = re.sub(r"^[-*]\s+|^\d+\.\s+", "", line)
        units.extend(s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z])", line) if s.strip())
    return [u.replace("**", "") for u in units if len(u) > 12 and not u.startswith("```")]


def extractive_answer(question: str, hits: list[Hit], embedder_name: str = "", idf: dict[str, float] | None = None) -> DocumentAnswer:
    sources = [Source(i + 1, hit) for i, hit in enumerate(hits[:SOURCES_FOR_ANSWER])]
    if not is_covered(question, hits, embedder_name, idf):
        return DocumentAnswer(question, NOT_COVERED, False, "extractive", sources)

    terms = set(words(question))
    scored = []
    for source in sources:
        for unit in _units(source.hit.chunk.text):
            overlap = len(terms & set(words(unit)))
            if overlap:
                scored.append((overlap - 0.15 * (source.number - 1), unit, source.number))
    if not scored:
        best = sources[0]
        units = _units(best.hit.chunk.text)[:SENTENCES]
        scored = [(1.0, unit, best.number) for unit in units]

    chosen, seen = [], set()
    ranked = sorted(scored, key=lambda item: -item[0])
    best = ranked[0][0]
    for score, unit, number in ranked:
        if score < 0.6 * best:
            break
        if unit not in seen:
            chosen.append(f"{unit.rstrip('.')}. [{number}]")
            seen.add(unit)
        if len(chosen) == SENTENCES:
            break
    cited = {int(n) for n in re.findall(r"\[(\d+)\]", " ".join(chosen))}
    return DocumentAnswer(question, " ".join(chosen), True, "extractive",
                          [s for s in sources if s.number in cited] or sources[:1], grounded=1.0)


SYSTEM = """You answer questions using only the numbered passages provided.
Rules:
- Every sentence must end with the number of the passage it comes from, like [2].
- Use only facts stated in the passages; copy numbers exactly.
- If the passages don't answer the question, reply exactly: The documents don't cover this question.
- Two to four sentences, no headings or lists."""


def check_answer(text: str, sources: list[Source]) -> tuple[float, list[str]]:
    """Share of sentences whose citations are valid and supported, and the problems found."""

    by_number = {s.number: s for s in sources}
    sentences = [s.strip() for s in re.split(r"(?<=[.!?\]])\s+(?=[A-Z])", text) if len(s.strip()) > 3]
    problems: list[str] = []
    good = 0
    for sentence in sentences:
        cited = [int(n) for n in re.findall(r"\[(\d+)\]", sentence)]
        if not cited:
            problems.append(f"No citation: \"{sentence[:80]}\"")
            continue
        unknown = [n for n in cited if n not in by_number]
        if unknown:
            problems.append(f"Cites passage {unknown[0]}, which was not retrieved.")
            continue
        passage = " ".join(by_number[n].hit.chunk.searchable() for n in cited)
        numbers = [n for n in NUMBER.findall(re.sub(r"\[\d+\]", "", sentence)) if n.replace(",", "") not in passage.replace(",", "")]
        if numbers:
            problems.append(f"Number {numbers[0]} is not in the cited passage.")
            continue
        content = set(words(re.sub(r"\[\d+\]", "", sentence)))
        if content and len(content & set(words(passage))) / len(content) < 0.4:
            problems.append(f"Not supported by the cited passage: \"{sentence[:80]}\"")
            continue
        good += 1
    return (good / len(sentences) if sentences else 0.0), problems


def model_answer(question: str, hits: list[Hit], provider: Provider, embedder_name: str = "", idf: dict[str, float] | None = None) -> DocumentAnswer:
    fallback = extractive_answer(question, hits, embedder_name, idf)
    sources = [Source(i + 1, hit) for i, hit in enumerate(hits[:SOURCES_FOR_ANSWER])]
    if not fallback.covered:
        return fallback

    passages = "\n\n".join(f"[{s.number}] {s.hit.chunk.section}\n{s.hit.chunk.text}" for s in sources)
    try:
        text = provider.chat([
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Passages:\n{passages}\n\nQuestion: {question.strip()}"},
        ]).strip()
    except LLMError as error:
        fallback.note = f"The language model could not be used ({error})."
        return fallback

    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    if text.startswith(NOT_COVERED[:30]):
        return DocumentAnswer(question, NOT_COVERED, False, "language model", sources)

    grounded, problems = check_answer(text, sources)
    if problems:
        fallback.note = "The language model's answer was not fully supported by the passages (" + problems[0] + "), so quoted passages are shown."
        fallback.problems = problems
        return fallback

    cited = {int(n) for n in re.findall(r"\[(\d+)\]", text)}
    return DocumentAnswer(question, text, True, "language model", [s for s in sources if s.number in cited], grounded=grounded)
