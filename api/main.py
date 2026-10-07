"""
AI Data Scientist Platform: HTTP API.

    python -m api                      # http://127.0.0.1:8000, docs at /docs
    python -m uvicorn api.main:app     # same app with uvicorn options

Every capability of the dashboard as typed endpoints: upload a dataset,
inspect schema and quality, ask questions, analyse a target, train and
save models, score new rows, plan contacts, forecast, segment, explain
changes and compute KPIs. Long work runs as a background job: start it
with POST, then poll GET /datasets/{id}/jobs/{job_id}.

Optional settings (environment variables):
    AIDS_API_KEY        if set, every request needs the header X-API-Key
    AIDS_CORS_ORIGINS   comma-separated origins allowed by CORS (default *)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from core.forecast import DEFAULT_HORIZON, ROW_COUNT, SeriesSpec, default_aggregation, suggest_freq
from core.llm import provider_status
from core.kpi import DEFAULT_FOLDER as DOMAIN_FOLDER, ConfigError, evaluate_domain, load_domain, parse_domain
from core.modeling import list_models, load_model, save_model
from core.modeling.features import prepare_frame
from core.recommend import PlanSettings
from core.segment import SegmentSpec
from core.target_analysis import detect_targets, report_markdown
from core.why import ChangeSpec, investigate
from service.knowledge import knowledge
from service.session import DatasetBundle, DatasetStore

from . import results
from .convert import plain, table
from .schemas import (
    AskRequest,
    AskResponse,
    DatasetInfo,
    AgentInfo,
    AgentMessage,
    AgentStarted,
    DocumentQuestion,
    DocumentSearch,
    DocumentStatus,
    DomainInfo,
    ForecastRequest,
    GoalPlanOut,
    GoalRun,
    GoalSpec,
    GoalText,
    LlmStatus,
    NarrativeOut,
    Health,
    JobStarted,
    JobStatus,
    KpiRequest,
    ModelRequest,
    QualityIssue,
    RecommendRequest,
    ScoreRequest,
    ScoreResponse,
    SegmentRequest,
    TargetResponse,
    WhyRequest,
)

VERSION = "0.6.0"
store = DatasetStore(capacity=8)


def require_key(x_api_key: str | None = Header(default=None)) -> None:
    expected = os.environ.get("AIDS_API_KEY")
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="Missing or wrong X-API-Key header.")


app = FastAPI(
    title="AI Data Scientist Platform API",
    version=VERSION,
    description="Understand any dataset, ask questions, build models, plan, forecast, "
                "segment, explain changes and compute KPIs over HTTP.",
    dependencies=[Depends(require_key)],
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in os.environ.get("AIDS_CORS_ORIGINS", "*").split(",") if o.strip()],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(ValueError)
async def value_error(_, error: ValueError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": str(error)})


def bundle_or_404(dataset_id: str) -> DatasetBundle:
    bundle = store.get(dataset_id)
    if bundle is None:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found (upload it again).")
    return bundle


def sheet_or_400(bundle: DatasetBundle, sheet: str | None) -> str | None:
    if sheet is None:
        return bundle.sheet
    if sheet not in bundle.sheets:
        raise HTTPException(status_code=400, detail=f"Sheet '{sheet}' not found; sheets: {bundle.sheets}.")
    return sheet


def started(bundle: DatasetBundle, job_id: str) -> JobStarted:
    return JobStarted(job_id=job_id, status="running", status_url=f"/datasets/{bundle.id}/jobs/{job_id}")


# ----------------------------------------------------------------------
# Service and datasets
# ----------------------------------------------------------------------

@app.get("/health", response_model=Health, tags=["service"])
def health() -> Health:
    return Health(status="ok", version=VERSION, datasets=len(store._items))


@app.post("/datasets", response_model=DatasetInfo, tags=["datasets"])
async def upload_dataset(file: UploadFile = File(...), sheet: str | None = Form(None)) -> DatasetInfo:
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    bundle = store.load(content, file.filename or "upload.csv", sheet)
    return dataset_info(bundle)


@app.get("/datasets", response_model=list[DatasetInfo], tags=["datasets"])
def list_datasets() -> list[DatasetInfo]:
    return [dataset_info(bundle, columns=False) for bundle in list(store._items.values())]


@app.get("/datasets/{dataset_id}", response_model=DatasetInfo, tags=["datasets"])
def get_dataset(dataset_id: str) -> DatasetInfo:
    return dataset_info(bundle_or_404(dataset_id))


def dataset_info(bundle: DatasetBundle, columns: bool = True) -> DatasetInfo:
    records = []
    if columns:
        for record in bundle.schema.to_records():
            records.append({
                "column": record["column"], "role": record["role"], "kind": record["kind"],
                "unique": record["unique"], "missing_pct": record["missing_pct"],
                "sample_values": plain(record["sample_values"]),
            })
    return DatasetInfo(
        id=bundle.id, name=bundle.name, sheet=bundle.sheet, sheets=bundle.sheets,
        rows=len(bundle.df), columns=bundle.df.shape[1], primary_time=bundle.schema.primary_time,
        column_info=records,
    )


@app.get("/datasets/{dataset_id}/quality", response_model=list[QualityIssue], tags=["datasets"])
def quality(dataset_id: str) -> list[QualityIssue]:
    return [
        QualityIssue(severity=issue["severity"], message=issue["message"], column=issue.get("column"))
        for issue in bundle_or_404(dataset_id).quality()
    ]


@app.get("/datasets/{dataset_id}/narrative", response_model=NarrativeOut, tags=["datasets"])
def narrative(dataset_id: str, kind: str = Query("dataset", pattern="^(dataset|model)$"), use_model: bool = False) -> NarrativeOut:
    """A plain-language summary from computed facts; with use_model, a language model's rewrite
    whose numbers are all checked against the facts (else the template is returned with a note)."""

    bundle = bundle_or_404(dataset_id)
    if kind == "model" and bundle.latest_model is None:
        raise HTTPException(status_code=404, detail="No model trained on this dataset yet.")
    sheet = bundle.facts(kind)
    text = bundle.narrative(kind, use_model)
    return NarrativeOut(kind=kind, facts=sheet.to_dict()["facts"], **text.to_dict())


@app.get("/datasets/{dataset_id}/insights", tags=["datasets"])
def insights(dataset_id: str, top: int = Query(10, ge=1, le=50)) -> list[dict[str, Any]]:
    keep = ("pattern_type", "description", "explanation", "evidence", "interestingness_score")
    return [
        {key: plain(insight.get(key)) for key in keep if key in insight}
        for insight in bundle_or_404(dataset_id).insights()[:top]
    ]


# ----------------------------------------------------------------------
# Goal box (free text -> validated plan -> run)
# ----------------------------------------------------------------------

@app.get("/llm/status", response_model=LlmStatus, tags=["goals"])
def llm_status(refresh: bool = False) -> LlmStatus:
    return LlmStatus(**provider_status(refresh))


@app.get("/datasets/{dataset_id}/goals/suggestions", response_model=list[GoalPlanOut], tags=["goals"])
def goal_suggestions(dataset_id: str) -> list[GoalPlanOut]:
    return [GoalPlanOut(**plain(plan.to_dict())) for plan in bundle_or_404(dataset_id).goal_suggestions()]


@app.post("/datasets/{dataset_id}/goals/interpret", response_model=GoalPlanOut, tags=["goals"])
def interpret_goal(dataset_id: str, request: GoalText) -> GoalPlanOut:
    """Read a goal in plain words: what will run, what was assumed, and any questions."""

    return GoalPlanOut(**plain(bundle_or_404(dataset_id).interpret_goal(request.text).to_dict()))


@app.post("/datasets/{dataset_id}/goals/check", response_model=GoalPlanOut, tags=["goals"])
def check_goal(dataset_id: str, spec: GoalSpec) -> GoalPlanOut:
    """Validate an edited goal (e.g. answers to the questions) and fill its defaults."""

    return GoalPlanOut(**plain(bundle_or_404(dataset_id).check_goal(spec.model_dump()).to_dict()))


@app.post("/datasets/{dataset_id}/goals/run", response_model=GoalRun, tags=["goals"])
def run_goal(dataset_id: str, spec: GoalSpec) -> GoalRun:
    """Run a complete goal: background job (poll status_url) or a direct answer."""

    bundle = bundle_or_404(dataset_id)
    started_goal = bundle.run_goal(spec.model_dump())
    if "entry" in started_goal:
        return GoalRun(tab=started_goal["tab"], answer=_ask_response(bundle, started_goal["entry"]))
    job_id = started_goal["job_id"]
    return GoalRun(tab=started_goal["tab"], job_id=job_id, status_url=f"/datasets/{bundle.id}/jobs/{job_id}")


# ----------------------------------------------------------------------
# Agents
# ----------------------------------------------------------------------

@app.get("/agents", response_model=list[AgentInfo], tags=["agents"])
def agents() -> list[AgentInfo]:
    from service.agents import AGENTS

    return [AgentInfo(name=key, label=label, tools=tools) for key, (label, _, tools) in AGENTS.items()]


@app.post("/datasets/{dataset_id}/agents/{agent}", response_model=AgentStarted, tags=["agents"])
def agent_turn(dataset_id: str, agent: str, request: AgentMessage) -> AgentStarted:
    """One turn of a conversation, run in the background; the job result is the answer with every step."""

    from service.agents import start_agent_job

    bundle = bundle_or_404(dataset_id)
    job_id, conversation_id = start_agent_job(bundle, agent, request.message, request.conversation_id, engine=request.engine)
    return AgentStarted(job_id=job_id, status_url=f"/datasets/{bundle.id}/jobs/{job_id}", conversation_id=conversation_id)


# ----------------------------------------------------------------------
# Documents (retrieval-augmented answers with citations)
# ----------------------------------------------------------------------

@app.get("/documents", response_model=DocumentStatus, tags=["documents"])
def documents_status(refresh: bool = False) -> DocumentStatus:
    index = knowledge.refresh() if refresh else knowledge.index()
    return DocumentStatus(**index.status())


@app.post("/documents/search", tags=["documents"])
def documents_search(request: DocumentSearch) -> list[dict[str, Any]]:
    """Ranked passages: hybrid (BM25 + vectors, fused by reciprocal rank), keyword or vector."""

    return [plain(hit.to_dict()) for hit in knowledge.search(request.query, request.k, request.mode)]


@app.post("/documents/ask", tags=["documents"])
def documents_ask(request: DocumentQuestion) -> dict[str, Any]:
    """An answer with numbered sources, or covered=false when the documents don't answer it."""

    return plain(knowledge.ask(request.question, request.use_model).to_dict())


@app.post("/documents", response_model=DocumentStatus, tags=["documents"])
async def documents_add(file: UploadFile = File(...)) -> DocumentStatus:
    knowledge.add_document(file.filename or "document.md", await file.read())
    return DocumentStatus(**knowledge.index().status())


@app.post("/documents/reindex", response_model=DocumentStatus, tags=["documents"])
def documents_reindex() -> DocumentStatus:
    return DocumentStatus(**knowledge.index(rebuild=True).status())


# ----------------------------------------------------------------------
# Ask
# ----------------------------------------------------------------------

@app.post("/datasets/{dataset_id}/ask", response_model=AskResponse, tags=["ask"])
def ask(dataset_id: str, request: AskRequest) -> AskResponse:
    bundle = bundle_or_404(dataset_id)
    return _ask_response(bundle, bundle.ask(request.question.strip()))


def _ask_response(bundle: DatasetBundle, entry: dict) -> AskResponse:
    chart = bundle.charts[entry["chart_key"]].to_dict() if entry.get("chart_key") else None
    return AskResponse(
        question=entry["question"],
        answer=entry["answer"],
        notice=entry.get("notice"),
        error=entry["error"],
        table=table(entry["table"], limit=200) if isinstance(entry.get("table"), pd.DataFrame) else None,
        chart=plain(chart),
        plan=plain(entry.get("plan")),
        by_model=bool(entry.get("by_model")),
    )


# ----------------------------------------------------------------------
# Target analysis and models
# ----------------------------------------------------------------------

@app.get("/datasets/{dataset_id}/target", response_model=TargetResponse, tags=["models"])
def target(dataset_id: str, column: str | None = None) -> TargetResponse:
    bundle = bundle_or_404(dataset_id)
    if column is None:
        candidates = detect_targets(bundle.df, bundle.schema)
        if not candidates:
            raise HTTPException(status_code=400, detail="No target column detected; pass ?column=.")
        column = candidates[0].column
    report = bundle.target_report(column)

    return TargetResponse(
        target=report.target, kind=report.kind, positive_label=plain(report.positive_label),
        rows=report.n_rows, labelled=report.n_labelled, positive_rate=plain(report.positive_rate),
        findings=[{"severity": f.severity, "title": f.title, "detail": f.detail, "column": f.column} for f in report.findings],
        split_method=report.split.method if report.split else None,
        split_summary=report.split.summary if report.split else None,
        split=table(report.split.table) if report.split and report.split.table is not None else None,
        segments=[
            {"column": s.column, "effect": plain(s.cramers_v), "p_value": plain(s.p_value),
             "highest": plain(list(s.highest)), "lowest": plain(list(s.lowest))}
            for s in report.segments
        ],
        features=table(pd.DataFrame([
            {"feature": f.column, "auc": f.auc, "correlation": f.correlation,
             "strength": f.strength, "direction": f.direction}
            for f in report.features
        ], columns=["feature", "auc", "correlation", "strength", "direction"])),
        excluded=[{"column": c, "reason": r} for c, r in report.excluded],
        markdown=report_markdown(report, bundle.name),
    )


@app.post("/datasets/{dataset_id}/models", response_model=JobStarted, tags=["models"])
def train_model(dataset_id: str, request: ModelRequest) -> JobStarted:
    bundle = bundle_or_404(dataset_id)
    target_column = request.target
    if target_column is None:
        from core.modeling import target_options

        options = target_options(bundle.df, bundle.schema, request.goal_type)
        if not options:
            raise HTTPException(status_code=400, detail="No column fits this goal type.")
        target_column = options[0]
    return started(bundle, bundle.start_model_job(request.goal_type, target_column))


@app.get("/datasets/{dataset_id}/models/latest", tags=["models"])
def latest_model(dataset_id: str, rows: int = Query(results.ROWS, ge=1, le=10000)) -> dict[str, Any]:
    bundle = bundle_or_404(dataset_id)
    if bundle.latest_model is None:
        raise HTTPException(status_code=404, detail="No model trained on this dataset yet.")
    return results.model_result(bundle.latest_model, rows)


@app.post("/datasets/{dataset_id}/models/save", tags=["models"])
def save_latest(dataset_id: str) -> dict[str, Any]:
    bundle = bundle_or_404(dataset_id)
    if bundle.latest_model is None:
        raise HTTPException(status_code=404, detail="No model trained on this dataset yet.")
    name = bundle.name if not bundle.sheet else f"{bundle.name} ({bundle.sheet})"
    return plain(save_model(bundle.latest_model, bundle.df, name))


@app.get("/models", tags=["models"])
def saved_models() -> list[dict[str, Any]]:
    return plain(list_models())


@app.post("/models/{name}/score", response_model=ScoreResponse, tags=["models"])
def score(name: str, request: ScoreRequest) -> ScoreResponse:
    try:
        saved = load_model(name)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Saved model '{name}' not found.")
    rows = pd.DataFrame(request.rows)
    missing = [c for c in saved["numeric"] + saved["categorical"] if c not in rows.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Rows are missing columns: {missing}.")
    X = prepare_frame(rows, saved["numeric"], saved["categorical"])
    pipeline = saved["pipeline"]
    values = pipeline.predict_proba(X)[:, 1] if saved["task"] == "classification" else pipeline.predict(X)
    return ScoreResponse(model=name, task=saved["task"], predictions=[plain(float(v)) for v in np.asarray(values)])


# ----------------------------------------------------------------------
# Jobs
# ----------------------------------------------------------------------

@app.get("/datasets/{dataset_id}/jobs/{job_id}", response_model=JobStatus, tags=["jobs"])
def job_status(dataset_id: str, job_id: str, rows: int = Query(results.ROWS, ge=1, le=10000)) -> JobStatus:
    bundle = bundle_or_404(dataset_id)
    job = bundle.model_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    result = None
    if job["status"] == "done":
        view = results.any_result(job["result"])
        if view and "scored" in view:
            view = results.model_result(job["result"], rows)
        result = view
    return JobStatus(job_id=job_id, status=job["status"], messages=list(job["messages"]),
                     error=job["error"], result=result)


# ----------------------------------------------------------------------
# Recommend, forecast, segments
# ----------------------------------------------------------------------

@app.post("/datasets/{dataset_id}/recommend", response_model=JobStarted, tags=["recommend"])
def recommend(dataset_id: str, request: RecommendRequest) -> JobStarted:
    bundle = bundle_or_404(dataset_id)
    sheets = bundle.interaction_sheets()
    sheet = request.sheet if request.sheet is not None else (sheets[0] if sheets else None)
    if request.sheet is not None:
        sheet_or_400(bundle, request.sheet)
    if not sheets and request.sheet is None:
        raise HTTPException(status_code=400, detail="No interaction table found in this dataset.")

    _, defaults = bundle.interaction_options(sheet)
    group = "auto" if request.group is None else (None if request.group.lower() == "none" else request.group)
    roles = bundle.interaction_roles(
        sheet,
        user=request.user or defaults.user,
        item=request.item or defaults.item,
        time=request.time or defaults.time,
        outcome=request.outcome or defaults.outcome,
        success_values=request.success_values if request.success_values else (
            defaults.success_values if (request.outcome in (None, defaults.outcome)) else None
        ),
        group=group,
    )
    settings = PlanSettings(
        start=pd.Timestamp(request.start) if request.start else None,
        days=request.days,
        capacity=request.capacity,
        min_gap_days=request.min_gap_days,
        period=request.period,
        one_owner=request.one_owner,
    )
    return started(bundle, bundle.start_recommend_job(sheet, roles, settings))


@app.get("/datasets/{dataset_id}/recommend/latest", tags=["recommend"])
def latest_plan(dataset_id: str, user: str | None = None, rows: int = Query(results.ROWS, ge=1, le=100000)) -> dict[str, Any]:
    bundle = bundle_or_404(dataset_id)
    if bundle.latest_recommendation is None:
        raise HTTPException(status_code=404, detail="No plan built on this dataset yet.")
    return results.recommend_result(bundle.latest_recommendation, rows, user)


@app.post("/datasets/{dataset_id}/forecast", response_model=JobStarted, tags=["forecast"])
def forecast(dataset_id: str, request: ForecastRequest) -> JobStarted:
    bundle = bundle_or_404(dataset_id)
    sheet = sheet_or_400(bundle, request.sheet)
    options = bundle.forecast_options(sheet)
    frame, schema, _, _ = bundle.interaction_context(sheet)
    if not options["dates"]:
        raise HTTPException(status_code=400, detail="This table has no date column.")

    time_column = request.time or options["dates"][0]
    measure = request.measure or options["measures"][0]
    freq = request.freq or suggest_freq(frame[time_column])
    aggregation = "count" if measure == ROW_COUNT else (request.aggregation or default_aggregation(schema, measure))
    spec = SeriesSpec(time=time_column, measure=measure, aggregation=aggregation, group=request.group,
                      freq=freq, horizon=request.horizon or DEFAULT_HORIZON[freq])
    return started(bundle, bundle.start_forecast_job(sheet, spec))


@app.post("/datasets/{dataset_id}/segments", response_model=JobStarted, tags=["segments"])
def segments(dataset_id: str, request: SegmentRequest) -> JobStarted:
    bundle = bundle_or_404(dataset_id)
    sheet = sheet_or_400(bundle, request.sheet)
    frame, _, _, _ = bundle.interaction_context(sheet)

    if request.unit == "rows":
        unit = None
    elif request.unit:
        unit = request.unit
    else:
        units = bundle.segment_options(sheet, None)["units"]
        unit = units[0] if units and len(frame) / frame[units[0]].nunique() >= 2 else None

    spec = SegmentSpec(unit=unit, features=request.features, method=request.method, k=request.k,
                       band_measure=request.band_measure, anomaly_share=request.anomaly_share)
    return started(bundle, bundle.start_segment_job(sheet, spec))


# ----------------------------------------------------------------------
# Why and KPIs (fast: answered directly)
# ----------------------------------------------------------------------

@app.post("/datasets/{dataset_id}/why", tags=["why"])
def why(dataset_id: str, request: WhyRequest) -> dict[str, Any]:
    bundle = bundle_or_404(dataset_id)
    sheet = sheet_or_400(bundle, request.sheet)
    options = bundle.why_options(sheet)
    frame, schema, _, _ = bundle.interaction_context(sheet)
    if not options["dates"]:
        raise HTTPException(status_code=400, detail="This table has no date column.")

    time_column = request.time or options["dates"][0]
    measure = request.measure or options["measures"][0]
    aggregation = "count" if measure == ROW_COUNT else (request.aggregation or default_aggregation(schema, measure))
    spec = ChangeSpec(
        time=time_column, measure=measure, aggregation=aggregation,
        freq=request.freq or suggest_freq(frame[time_column]), compare=request.compare,
        dimensions=request.dimensions if request.dimensions is not None else options["dimensions"],
    )
    if request.attention and request.attention.lower() == "none":
        entity = None
    else:
        entity = request.attention or (options["entities"][0] if options["entities"] else None)

    return results.investigation_result(investigate(frame, spec, entity, request.higher_is_better))


@app.get("/domains", response_model=list[DomainInfo], tags=["kpis"])
def domains(dataset_id: str | None = None) -> list[DomainInfo]:
    if dataset_id:
        bundle = bundle_or_404(dataset_id)
        found = bundle.kpi_domains()
    else:
        found = []
        for path in sorted(Path(DOMAIN_FOLDER).glob("*.y*ml")):
            try:
                config = load_domain(path)
                found.append({"path": str(path), "name": config.name, "usable": 0, "total": len(config.kpis), "error": ""})
            except (ConfigError, yaml.YAMLError) as error:
                found.append({"path": str(path), "name": path.stem, "usable": 0, "total": 0, "error": str(error)})
    return [DomainInfo(name=d["name"], file=Path(d["path"]).stem, usable=d["usable"], total=d["total"], error=d["error"]) for d in found]


@app.post("/datasets/{dataset_id}/kpis", tags=["kpis"])
def kpis(dataset_id: str, request: KpiRequest) -> dict[str, Any]:
    bundle = bundle_or_404(dataset_id)

    if request.domain_yaml:
        try:
            config = parse_domain(yaml.safe_load(request.domain_yaml) or {})
        except yaml.YAMLError as error:
            raise HTTPException(status_code=400, detail=f"Invalid YAML: {error}")
    elif request.domain:
        path = Path(DOMAIN_FOLDER) / f"{Path(request.domain).stem}.yaml"
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"Domain file '{request.domain}' not found.")
        config = load_domain(path)
    else:
        raise HTTPException(status_code=400, detail="Pass 'domain' (a file name) or 'domain_yaml'.")

    report = evaluate_domain(config, bundle.kpi_table, request.freq, request.group_role)
    return {
        "domain": config.name,
        "current_period": plain(report.current_period),
        "previous_period": plain(report.previous_period),
        "computed_at": report.computed_at,
        "kpis": table(report.summary.drop(columns=["_id"])),
        "by_group": table(report.by_group),
        "trend": table(report.trend),
        "unavailable": table(report.unavailable),
    }


@app.get("/datasets/{dataset_id}/kpis/starter", tags=["kpis"])
def kpi_starter(dataset_id: str) -> dict[str, str]:
    return {"yaml": bundle_or_404(dataset_id).kpi_starter()}
