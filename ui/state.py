"""
Server-side dataset state for the dashboard.

Dash stores live in the browser as JSON, so the dataframe and every
derived artefact (schema, quality report, recommendations, insights,
NL query objects, chart registry) stay here. The browser only holds the
dataset id and the ids of pinned charts.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from core.data_quality import run_quality_checks
from core.eda_pipeline import run_insight_pipeline
from core.loader import (
    default_sheet,
    list_sheets,
    load_dataset_from_bytes,
)
from core.NLP.answer_generator import AnswerGenerator
from core.NLP.query_engine import QueryEngine
from core.NLP.query_parser import QueryParser, code_key
from core.NLP.query_plan import QueryPlanner
from core.schema_inference import DatasetSchema, infer_schema, name_tokens, same_word
from core.modeling import GoalSpec, ModelResult, build_model, propose_goal
from core.target_analysis import TargetReport, analyze_target, detect_targets
from core.workbook import Workbook
from visualization import (
    ChartEngine,
    ChartRecommender,
    ChartRenderer,
    ChartSpec,
    ManualChartBuilder,
    insight_to_chart_spec,
    query_to_chart_spec,
)


chart_engine = ChartEngine()
generator = AnswerGenerator()
chart_renderer = ChartRenderer()
chart_recommender = ChartRecommender()
manual_builder = ManualChartBuilder(recommender=chart_recommender)

# Rows kept for the table under an answer (the answer text has the count).
MAX_ANSWER_ROWS = 1000


@dataclass
class RenderedChart:
    key: str
    spec: ChartSpec
    figure: Any
    data: list[dict[str, Any]]
    metadata: dict[str, Any]


@dataclass
class DatasetBundle:
    id: str
    name: str
    content: bytes
    sheets: list[str]
    sheet: str | None
    df: pd.DataFrame
    schema: DatasetSchema

    charts: dict[str, ChartSpec] = field(default_factory=dict)
    ask_history: list[dict[str, Any]] = field(default_factory=list)

    _quality: list[dict[str, Any]] | None = None
    _recommendations: list | None = None
    _insights: list[dict[str, Any]] | None = None
    # Answer charts computed on a (linked) sheet: chart key -> sheet.
    chart_sheets: dict[str, str] = field(default_factory=dict)

    _workbook: Workbook | None = None
    _target_reports: dict[str, TargetReport] = field(default_factory=dict)
    _model_jobs: dict[str, dict[str, Any]] = field(default_factory=dict)
    latest_model: ModelResult | None = None
    _contexts: dict[str, "AskContext"] = field(default_factory=dict)
    _lock: threading.RLock = field(default_factory=threading.RLock)

    # ------------------------------------------------------------------
    # Cached analysis
    # ------------------------------------------------------------------

    def quality(self) -> list[dict[str, Any]]:
        with self._lock:
            if self._quality is None:
                self._quality = run_quality_checks(self.df, self.schema)
            return self._quality

    def recommendations(self, max_charts: int = 6) -> list:
        with self._lock:
            if self._recommendations is None:
                self._recommendations = (
                    chart_recommender.recommend_default_charts(
                        self.df, max_charts=max_charts, schema=self.schema,
                    )
                )
            return self._recommendations

    def insights(self, top_n: int = 10) -> list[dict[str, Any]]:
        with self._lock:
            if self._insights is None:
                insights = run_insight_pipeline(
                    self.df, self.schema, top_n=top_n,
                )
                for insight in insights:
                    spec = insight_to_chart_spec(insight)
                    insight["chart_key"] = (
                        self.register(spec) if spec else None
                    )
                self._insights = insights
            return self._insights

    # ------------------------------------------------------------------
    # Target analysis (training tables)
    # ------------------------------------------------------------------

    def target_candidates(self) -> list:
        return detect_targets(self.df, self.schema)

    def target_report(self, target: str) -> TargetReport:
        with self._lock:
            if target not in self._target_reports:
                self._target_reports[target] = analyze_target(
                    self.df, self.schema, target,
                )
            return self._target_reports[target]

    # ------------------------------------------------------------------
    # Model building (runs in a background thread)
    # ------------------------------------------------------------------

    def model_goal(self, goal_type: str, target: str) -> GoalSpec:
        return propose_goal(
            self.df, self.schema, goal_type, target,
            report=self.target_report(target),
        )

    def start_model_job(self, goal_type: str, target: str) -> str:
        spec = self.model_goal(goal_type, target)
        job_id = uuid.uuid4().hex[:8]
        job: dict[str, Any] = {
            "id": job_id,
            "status": "running",
            "messages": [],
            "result": None,
            "error": None,
        }

        with self._lock:
            self._model_jobs[job_id] = job

        def run() -> None:
            try:
                result = build_model(self.df, self.schema, spec, progress=job["messages"].append)
                job["result"] = result
                self.latest_model = result
                job["status"] = "done"
            except Exception as error:  # noqa: BLE001 - reported in the UI
                job["error"] = str(error)
                job["status"] = "error"

        threading.Thread(target=run, daemon=True).start()
        return job_id

    def model_job(self, job_id: str | None) -> dict[str, Any] | None:
        if not job_id:
            return None
        with self._lock:
            return self._model_jobs.get(job_id)

    # ------------------------------------------------------------------
    # Charts
    # ------------------------------------------------------------------

    def register(self, spec: ChartSpec, sheet: str | None = None) -> str:
        """
        Store a spec and return a stable key for pin buttons. Charts
        from answers remember the sheet they were computed on.
        """

        payload = json.dumps(
            {"spec": spec.to_dict(), "sheet": sheet}, sort_keys=True, default=str,
        )
        key = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]

        with self._lock:
            self.charts.setdefault(key, spec)
            if sheet is not None:
                self.chart_sheets.setdefault(key, sheet)

        return key

    def render(self, spec_or_key: ChartSpec | str) -> RenderedChart:
        if isinstance(spec_or_key, str):
            spec = self.charts[spec_or_key]
            key = spec_or_key
        else:
            spec = spec_or_key
            key = self.register(spec)

        result = chart_engine.create_chart(self._chart_frame(key), spec)
        figure = chart_renderer.render(result)

        return RenderedChart(
            key=key,
            spec=spec,
            figure=figure,
            data=result.data,
            metadata=result.metadata,
        )

    def _chart_frame(self, key: str) -> pd.DataFrame:
        sheet = self.chart_sheets.get(key)
        if sheet is None:
            return self.df
        return self._ask_context(sheet).df

    def build_manual(self, **selection: Any) -> ChartSpec:
        return manual_builder.build(
            self.df, schema=self.schema, **selection,
        )

    # ------------------------------------------------------------------
    # Workbook (all sheets of an Excel file, linked by ID columns)
    # ------------------------------------------------------------------

    @property
    def workbook(self) -> Workbook | None:
        with self._lock:
            if self._workbook is None and len(self.sheets) > 1:
                self._workbook = Workbook(self.content, self.name, self.sheets)
            return self._workbook

    def _ask_context(self, sheet: str | None) -> AskContext:
        """
        The frame questions run on: the sheet plus the columns of the
        sheets it links to. Without a workbook, the loaded dataset.
        """

        with self._lock:
            key = sheet or ""

            if key not in self._contexts:
                workbook = self.workbook
                enriched = workbook.enriched(sheet) if workbook and sheet else None

                if enriched is None:
                    df, schema, sources = self.df, self.schema, {}
                else:
                    df = enriched.df
                    schema = infer_schema(df)
                    sources = enriched.sources

                self._contexts[key] = AskContext(
                    sheet=sheet,
                    df=df,
                    schema=schema,
                    sources=sources,
                    parser=QueryParser(df, schema=schema),
                    planner=QueryPlanner(df, schema=schema),
                    engine=QueryEngine(df, schema=schema),
                )

            return self._contexts[key]

    # ------------------------------------------------------------------
    # Natural-language questions
    # ------------------------------------------------------------------

    def ask(self, question: str) -> dict[str, Any]:
        entry: dict[str, Any] = {
            "id": uuid.uuid4().hex[:8],
            "question": question,
            "answer": None,
            "table": None,
            "chart_key": None,
            "error": None,
            "notice": None,
        }

        try:
            attempt = self._attempt(self.sheet, question)

            # The selected sheet can't answer: try the other sheets and
            # keep the one that resolves the most of the question.
            if not attempt.ok and self.workbook is not None:
                others = [
                    self._attempt(sheet, question)
                    for sheet in self.workbook.readable_sheets()
                    if sheet != self.sheet
                ]
                answered = [other for other in others if other.ok]
                if answered:
                    attempt = max(answered, key=lambda other: other.score)

            self._fill(entry, attempt)

        except Exception:  # noqa: BLE001 - a question must never break the page
            entry["error"] = "Something went wrong while answering this question."

        return self._remember(entry)

    def _attempt(self, sheet: str | None, question: str) -> Attempt:
        context = self._ask_context(sheet)
        attempt = Attempt(context=context)

        try:
            parsed = context.parser.parse(question)
        except (ValueError, TypeError, KeyError) as error:
            attempt.error = str(error)
            return attempt

        attempt.parsed = parsed

        # Answering without a value the question names ("MR01")
        # would silently answer a different question.
        if parsed.unresolved:
            return attempt

        try:
            attempt.plan = context.planner.create_plan(parsed)
            attempt.result = context.engine.execute(attempt.plan)
            attempt.answer = generator.generate(
                question, parsed, attempt.plan, attempt.result,
            )
        except (ValueError, TypeError, KeyError) as error:
            attempt.error = str(error)

        attempt.score = self._score(attempt, question)

        return attempt

    def _score(self, attempt: Attempt, question: str) -> tuple:
        """
        Prefer the sheet that uses more of the question (entity,
        filters, grouping), then one whose name the question mentions
        ("visited" ~ Visits), then the selected sheet.
        """

        parsed = attempt.parsed
        resolved = (
            len(parsed.filters)
            + (parsed.entity is not None)
            + (parsed.group_by is not None)
        )

        sheet = attempt.context.sheet or ""
        words = re.findall(r"[a-z]+", question.lower())
        named = any(
            same_word(word, token)
            for word in words
            for token in name_tokens(sheet)
        )

        return resolved, named, sheet == self.sheet

    def _fill(self, entry: dict[str, Any], attempt: Attempt) -> None:
        context = attempt.context

        if attempt.parsed is not None and attempt.parsed.unresolved and attempt.error is None:
            entry["notice"] = self._unresolved_notice(attempt.parsed.unresolved)
            return

        if attempt.error is not None:
            entry["error"] = attempt.error
            return

        result = attempt.result
        notes = []

        used = context.sources and _columns_used(attempt.plan, result)
        lookups = [
            sheet
            for sheet in dict.fromkeys(context.sources.get(column) for column in used or [])
            if sheet
        ]

        if context.sheet != self.sheet:
            source = f"Answered from the '{context.sheet}' sheet"
            if lookups:
                source += ", with details from " + _quoted(lookups)
            notes.append(source + ".")
        elif lookups:
            notes.append("Uses details from " + _quoted(lookups) + ".")

        entry["answer"] = " ".join([attempt.answer, *notes])

        # A requested chart shows its numbers under "View data".
        if isinstance(result, pd.DataFrame) and attempt.plan.operation != "CHART":
            entry["table"] = result.head(MAX_ANSWER_ROWS)

        spec = query_to_chart_spec(attempt.plan, context.schema)

        if spec is not None:
            sheet = context.sheet if self.workbook is not None else None
            try:
                chart_engine.create_chart(context.df, spec)
                entry["chart_key"] = self.register(spec, sheet=sheet)
            except (ValueError, TypeError, KeyError):
                entry["chart_key"] = None

    def _remember(self, entry: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self.ask_history.insert(0, entry)
            del self.ask_history[20:]
        return entry

    def _unresolved_notice(self, tokens: list[str]) -> str:
        """
        Explain values that no sheet could answer with, naming the
        sheets that contain them.
        """

        messages = []

        for token in tokens:
            found = self.locate_value(token)
            message = f"'{token}' isn't in this sheet"

            if found:
                places = ", ".join(
                    f"sheet '{sheet}' (column {column})"
                    for sheet, column in found[:3]
                )
                target = "that sheet" if len(found) == 1 else "one of those sheets"
                message += (
                    f". It appears in {places}. Switch to {target} "
                    f"with the sheet selector to ask about it."
                )
            else:
                message += " or in any other sheet of this file."

            messages.append(message)

        return " ".join(messages)

    def locate_value(self, token: str) -> list[tuple[str, str]]:
        """(sheet, column) pairs in other sheets that contain the value."""

        if self.workbook is None:
            return []

        target = code_key(token)
        text = token.strip().lower()
        found = []

        for sheet in self.workbook.readable_sheets():
            if sheet == self.sheet:
                continue

            frame = self.workbook.frame(sheet)

            for column in frame.columns:
                series = frame[column]

                if series.dtype.kind in "biufcmM":
                    continue

                values = series.dropna().astype(str).str.strip().unique()

                if any(
                    value.lower() == text
                    or (target is not None and code_key(value) == target)
                    for value in values
                ):
                    found.append((sheet, str(column)))
                    break

        return found


@dataclass
class AskContext:
    sheet: str | None
    df: pd.DataFrame
    schema: DatasetSchema
    # Column added from a linked sheet -> that sheet.
    sources: dict[str, str]
    parser: QueryParser
    planner: QueryPlanner
    engine: QueryEngine


@dataclass
class Attempt:
    context: AskContext
    parsed: Any = None
    plan: Any = None
    result: Any = None
    answer: str | None = None
    error: str | None = None
    score: tuple = ()

    @property
    def ok(self) -> bool:
        return (
            self.answer is not None
            and self.error is None
            and not (self.parsed is not None and self.parsed.unresolved)
        )


def _columns_used(plan, result) -> list[str]:
    columns = [plan.column, plan.group_by, plan.subject]
    columns += [item["column"] for item in plan.filters or []]
    if isinstance(result, pd.DataFrame):
        columns += list(result.columns)
    return [column for column in columns if column]


def _quoted(sheets: list[str]) -> str:
    names = [f"'{sheet}'" for sheet in sheets]
    if len(names) == 1:
        return f"the {names[0]} sheet"
    return "the " + ", ".join(names[:-1]) + f" and {names[-1]} sheets"


class DatasetStore:
    """Keeps the most recently loaded datasets in memory."""

    def __init__(self, capacity: int = 4):
        self.capacity = capacity
        self._items: OrderedDict[str, DatasetBundle] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, dataset_id: str | None) -> DatasetBundle | None:
        if not dataset_id:
            return None
        with self._lock:
            bundle = self._items.get(dataset_id)
            if bundle is not None:
                self._items.move_to_end(dataset_id)
            return bundle

    def load(
        self,
        content: bytes,
        filename: str,
        sheet: str | None = None,
    ) -> DatasetBundle:

        sheets = list_sheets(content, filename)

        if sheets and sheet not in sheets:
            sheet = default_sheet(content, filename)

        df = load_dataset_from_bytes(content, filename, sheet)

        bundle = DatasetBundle(
            id=uuid.uuid4().hex,
            name=filename,
            content=content,
            sheets=sheets,
            sheet=sheet,
            df=df,
            schema=infer_schema(df),
        )

        with self._lock:
            self._items[bundle.id] = bundle
            while len(self._items) > self.capacity:
                self._items.popitem(last=False)

        # Warm the caches so the Auto insights tab opens instantly.
        threading.Thread(
            target=_warm_up,
            args=(bundle,),
            daemon=True,
        ).start()

        return bundle


def _warm_up(bundle: DatasetBundle) -> None:
    try:
        bundle.quality()
        for recommendation in bundle.recommendations():
            bundle.register(recommendation.chart_spec)
        bundle.insights()
        # Load and link the other sheets so the first question is quick.
        bundle._ask_context(bundle.sheet)
    except Exception:  # noqa: BLE001 - tabs recompute and report errors
        pass


store = DatasetStore()
