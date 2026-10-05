"""
Rule-based reading of a goal: generic English cues plus column names.

    "forecast revenue by region for the next 6 months"
        task cues  -> forecast
        names      -> measure "Revenue", group "Region"
        time words -> freq "M", horizon 6

Cue words are plain English hints (forecast, why, segment, plan, rank,
predict ...), never domain vocabulary; columns are matched through their
name tokens, so "orders" finds "OrderId" and "regions" finds "Region".
Words that match no cue, column, sheet or category value are reported
as unknown: rules cannot map synonyms, which is when a language model
is asked.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from core.schema_inference import name_tokens, same_word

from .spec import MODEL_TASKS
from .view import DataView

# (pattern, weight): strong cues name the task outright.
TASK_CUES: dict[str, list[tuple[str, int]]] = {
    "forecast": [(r"\bforecast\w*", 3), (r"\bproject(ion|ed)?s?\b", 2), (r"\bfuture\b", 1),
                 (r"\bupcoming\b", 1), (r"\boutlook\b", 1), (r"\bahead\b", 1)],
    "why": [(r"\bwhy\b", 3), (r"\breasons?\b", 2), (r"\bcause[sd]?\b", 2), (r"\bdrivers?\b", 1),
            (r"\bexplain\w*", 1), (r"\bdrop(ped|s)?\b", 1), (r"\bfell\b", 1), (r"\bdeclin\w*", 1),
            (r"\bincreas\w*", 1), (r"\brose\b", 1), (r"\bwent (up|down)\b", 2), (r"\bchanged\b", 1)],
    "segment": [(r"\bsegment\w*", 3), (r"\bcluster\w*", 3), (r"\banomal\w*", 3), (r"\boutliers?\b", 3),
                (r"\bsimilar\b", 2), (r"\bpersonas?\b", 2), (r"\bunusual\b", 2), (r"\bgroup\w* .*\binto\b", 2)],
    "recommend": [(r"\brecommend\w*", 3), (r"\bplan(s|ning)?\b", 3), (r"\bschedul\w*", 3),
                  (r"\bnext best\b", 3), (r"\bwho should\b", 2), (r"\bwhich \w+ should\b", 2),
                  (r"\bassign\w*", 2), (r"\bitinerar\w*", 2), (r"\broutes?\b", 1)],
    "rank": [(r"\brank\w*", 3), (r"\bprioriti[sz]\w*", 3), (r"\bpriority\b", 2), (r"\bmost likely\b", 2),
             (r"\btop \d*\s*\w+ (to|most|likely)\b", 2), (r"\bscor(e|ing)\b", 1)],
    "classify": [(r"\bwill\b", 1), (r"\bwhether\b", 1), (r"\blikel(y|ihood)\b", 1), (r"\bprobabilit\w*", 2),
                 (r"\bchances?\b", 1), (r"\brisk\b", 1), (r"\bclassif\w*", 3), (r"\bpredict\w*", 1)],
    "regress": [(r"\bhow much\b", 1), (r"\bestimat\w*", 1), (r"\bpredict\w*", 1)],
    "ask": [(r"^\s*(list|show|give|display|find|count|what|which|who|how many|total)\b", 1)],
}

NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "a": 1, "an": 1,
}
UNIT_FREQ = {"day": "D", "week": "W", "month": "M", "quarter": "Q"}
FUTURE = re.compile(
    r"\b(next|coming|following|upcoming)\s+(?:(\d+|" + "|".join(NUMBER_WORDS) + r")\s+)?(day|week|month|quarter|year)s?\b"
)
EVERY = {"daily": "D", "weekly": "W", "monthly": "M", "quarterly": "Q"}
SPLIT = re.compile(r"\b(by|per|for each|for every|each|across|split by|broken down by)\s+$")

# Generic English words that carry no column meaning in a goal.
STOPWORDS = set("""
a an the and or of to for in on at by with from into over under than then that this these those there their
them they it its is are was were be been being do does did done have has had having can could would should will
shall may might must i we you me my our your us want wants wanted need needs like please help let lets make build
create get give show list find tell see run use using based each every all any some most more less many much few
which what who whom whose when where how why whether if so as also just only per about between across within
next last previous coming following upcoming new latest current past future time times period periods
day days week weeks month months quarter quarters year years daily weekly monthly quarterly yearly today
data dataset table tables sheet sheets column columns row rows record records value values number numbers count
total totals sum average mean model models prediction predictions predict likely high higher low lower best good
better top first going go goes work see look looking way ways thing things one two three four five six seven
eight nine ten eleven twelve kind type types level levels change changes changed
am as be he ok vs up eg ie
call calls calling contact contacts contacting meet meeting meetings reach reaching approach
""".split())


@dataclass
class Mention:
    column: str
    sheet: str | None
    position: int
    score: float          # share of the column's name words found in the text
    role: str | None
    unique: bool = False  # a key with (almost) one row per value: not a role like user or unit


@dataclass
class RuleReading:
    fields: dict[str, Any]
    scores: dict[str, int]
    mentions: list[Mention] = field(default_factory=list)
    unknown: list[str] = field(default_factory=list)
    ambiguous: list[str] = field(default_factory=list)
    soft: set[str] = field(default_factory=set)            # fields read from partial name matches
    sheet_hints: list = field(default_factory=list)        # tables named in the text

    @property
    def task(self) -> str | None:
        return self.fields.get("task")

    @property
    def confident(self) -> bool:
        return self.task is not None and not self.ambiguous and not self.unknown


def read_goal(text: str, view: DataView) -> RuleReading:
    lowered = " " + re.sub(r"\s+", " ", text.lower()).strip() + " "
    words = [(m.group(), m.start()) for m in re.finditer(r"[a-z]+", lowered)]

    scores, cue_spans = _task_scores(lowered)
    mentions = _mentions(words, view)
    fields: dict[str, Any] = {}

    _time_words(lowered, fields)
    _adjust_scores(lowered, scores, mentions, fields)

    best = max(scores.values()) if scores else 0
    leaders = sorted(task for task, score in scores.items() if score == best and score > 0)
    ambiguous: list[str] = []
    if len(leaders) == 1:
        fields["task"] = leaders[0]
    elif len(leaders) > 1:
        preferred = [t for t in leaders if t != "ask"]
        if set(preferred) == {"classify", "rank"}:
            fields["task"] = "rank"
        else:
            ambiguous = preferred or leaders

    soft = _assign_columns(fields, mentions, lowered)
    if fields.get("task") == "ask":
        fields["question"] = text.strip()

    unknown = _unknown_words(words, cue_spans, mentions, view)
    return RuleReading(fields=fields, scores=scores, mentions=mentions, unknown=unknown,
                       ambiguous=ambiguous, soft=soft, sheet_hints=_sheet_hints(words, view))


# ----------------------------------------------------------------------
# Task
# ----------------------------------------------------------------------

def _task_scores(lowered: str) -> tuple[dict[str, int], list[tuple[int, int]]]:
    scores: dict[str, int] = {}
    spans: list[tuple[int, int]] = []
    for task, cues in TASK_CUES.items():
        for pattern, weight in cues:
            for match in re.finditer(pattern, lowered):
                scores[task] = scores.get(task, 0) + weight
                spans.append(match.span())
    return scores, spans


def _adjust_scores(lowered: str, scores: dict[str, int], mentions: list[Mention], fields: dict) -> None:
    """Disambiguate 'predict' and friends with the roles of the named columns."""

    roles = {m.role for m in mentions}
    predictive = bool(re.search(r"\b(predict\w*|will|likel\w*|estimat\w*|how much)\b", lowered))
    future = "horizon" in fields

    if predictive or future:
        if future and "measure" in roles and "binary" not in roles:
            scores["forecast"] = scores.get("forecast", 0) + 3
        elif "binary" in roles:
            scores["classify"] = scores.get("classify", 0) + 2
            scores.pop("regress", None)
        elif "measure" in roles and not future:
            scores["regress"] = scores.get("regress", 0) + 2
            scores.pop("classify", None)
        else:
            scores.pop("regress", None)

    if scores.get("rank"):
        scores.pop("classify", None)
    if scores.get("recommend", 0) >= 3:
        for task in ("classify", "regress", "forecast"):
            scores.pop(task, None)
    if any(scores.get(t, 0) >= 2 for t in scores if t != "ask"):
        scores.pop("ask", None)


def _time_words(lowered: str, fields: dict) -> None:
    match = FUTURE.search(lowered)
    if match:
        amount = match.group(2)
        n = int(amount) if amount and amount.isdigit() else NUMBER_WORDS.get(amount or "", 1)
        unit = match.group(3)
        if unit == "year":
            fields["freq"], fields["horizon"] = "M", 12 * n
        else:
            fields["freq"], fields["horizon"] = UNIT_FREQ[unit], n
        if unit == "month" and n == 1:
            fields["period"] = "month"
        elif unit == "day":
            fields["period"], fields["days"] = "days", n
        elif unit == "week":
            fields["period"], fields["days"] = "days", 5 * n

    for word, freq in EVERY.items():
        if re.search(rf"\b{word}\b", lowered):
            fields["freq"] = freq

    if re.search(r"\b(last year|year on year|year over year|yoy|same \w+ last year)\b", lowered):
        fields["compare"] = "year"


# ----------------------------------------------------------------------
# Columns
# ----------------------------------------------------------------------

def _content_tokens(column: str) -> list[str]:
    from core.schema_inference import ENTITY_KEY_TOKENS

    tokens = name_tokens(column)
    content = [t for t in tokens if t not in ENTITY_KEY_TOKENS and t not in STOPWORDS and not t.isdigit()]
    return content or [t for t in tokens if t not in ENTITY_KEY_TOKENS] or tokens


def _mentions(words: list[tuple[str, int]], view: DataView) -> list[Mention]:
    """Columns whose name words appear in the text, the best match per column."""

    found: dict[tuple[str, Any], Mention] = {}
    # Two-letter words count only as exact name words (e.g. an abbreviation used as a key name).
    usable = [(w, p) for w, p in words if len(w) >= 2 and w not in STOPWORDS]

    for sheet in view.sheets:
        try:
            frame, schema, _, _ = view.context(sheet)
        except (ValueError, KeyError, TypeError):
            continue
        for column in view.native_columns(sheet):
            tokens = _content_tokens(column)
            hits = [p for t in tokens for w, p in usable if same_word(w, t)]
            matched = {t for t in tokens if any(same_word(w, t) for w, _ in usable)}
            if not matched:
                continue
            score = len(matched) / len(tokens)
            role = schema.role_of(column)
            if role == "time" and score < 1:
                continue
            unique = role == "identifier" and frame[column].nunique() >= 0.9 * len(frame)
            mention = Mention(column, sheet, min(hits), score, role, unique)
            key = (column, sheet)
            if key not in found or score > found[key].score:
                found[key] = mention

    # Keep, for each place in the text, the columns that match it best.
    best: dict[int, float] = {}
    for mention in found.values():
        best[mention.position] = max(best.get(mention.position, 0), mention.score)
    return sorted(
        (m for m in found.values() if m.score >= best[m.position]),
        key=lambda m: (m.position, -m.score),
    )


def _assign_columns(fields: dict, mentions: list[Mention], lowered: str) -> set[str]:
    """Put named columns into the task's fields; returns fields read from partial matches."""

    task = fields.get("task")
    if not mentions or task in (None, "ask"):
        return set()

    chosen: dict[str, Mention] = {}

    def ranked(roles: set[str], repeated: bool = False, exclude: tuple = ()) -> list[Mention]:
        found = [m for m in mentions if m.role in roles and m.column not in exclude and not (repeated and m.unique)]
        unique: dict[str, Mention] = {}
        for mention in sorted(found, key=lambda m: (-m.score, m.position)):
            unique.setdefault(mention.column, mention)
        return list(unique.values())

    def take(field: str, roles: set[str], repeated: bool = False, exclude: tuple = ()) -> None:
        found = ranked(roles, repeated, exclude)
        if found:
            chosen[field] = found[0]

    def split(mention: Mention) -> bool:
        return bool(SPLIT.search(lowered[: mention.position]))

    if task in MODEL_TASKS:
        take("target", {"measure"} if task == "regress" else {"binary"})
    elif task in ("forecast", "why"):
        take("measure", {"measure"})
        if "measure" not in chosen:
            take("measure", {"identifier"})
        take("time", {"time"})
        measure = chosen["measure"].column if "measure" in chosen else None
        if task == "forecast":
            groups = [m for m in ranked({"dimension", "identifier"}, True, (measure,)) if split(m)]
            if groups:
                chosen["group"] = groups[0]
        else:
            dimensions = [m.column for m in ranked({"dimension"}, False, (measure,))]
            if dimensions:
                fields["dimensions"] = list(dict.fromkeys(dimensions))
            take("attention", {"identifier"}, True, (measure,))
    elif task == "segment":
        take("unit", {"identifier"}, True)
        features = [m.column for m in ranked({"measure", "dimension", "binary"})]
        if len(features) >= 2:
            fields["features"] = list(dict.fromkeys(features))
    elif task == "recommend":
        keys = ranked({"identifier", "dimension"}, True)
        owners = [
            m for m in keys
            if split(m) or re.search(
                r"\b(for|of|by)\s+(each|every|all|the|a|an)?\s*" + re.escape(_mention_word(lowered, m)), lowered
            )
        ]
        if owners:
            chosen["user"] = owners[0]
        owner = owners[0].column if owners else None
        items = [m for m in keys if m.column != owner]
        if items:
            chosen["item"] = items[0]
        if not owners and len(items) >= 2:
            chosen["user"] = items[1]
        take("time", {"time"})

    for field, mention in chosen.items():
        fields[field] = mention.column
    return {field for field, mention in chosen.items() if mention.score < 1}


def _sheet_hints(words, view: DataView) -> list:
    hints = []
    for sheet in view.sheets:
        if not sheet:
            continue
        tokens = [t for t in name_tokens(sheet) if t not in STOPWORDS]
        if tokens and any(same_word(w, t) for w, _ in words if len(w) >= 3 for t in tokens):
            hints.append(sheet)
    return hints


def _mention_word(lowered: str, mention: Mention) -> str:
    match = re.match(r"[a-z]+", lowered[mention.position:])
    return match.group() if match else column.lower()


def _unknown_words(words, cue_spans, mentions: list[Mention], view: DataView) -> list[str]:
    covered = {p for start, end in cue_spans for p in range(start, end)}
    known_positions = {m.position for m in mentions}
    column_tokens = {t for m in mentions for t in name_tokens(m.column)}
    sheet_tokens = {t for s in view.sheets if s for t in name_tokens(s)}

    unknown = []
    for word, position in words:
        if len(word) < 4 or word in STOPWORDS or position in covered or position in known_positions:
            continue
        if any(same_word(word, t) for t in column_tokens | sheet_tokens):
            continue
        if any(word in view.value_words(sheet) for sheet in view.sheets):
            continue
        if word not in unknown:
            unknown.append(word)
    return unknown
