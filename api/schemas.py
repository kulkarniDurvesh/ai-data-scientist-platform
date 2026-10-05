"""
Request and response models.

They document every endpoint in the OpenAPI schema (/docs, /openapi.json),
so clients such as the .NET SFA app can generate a typed client. Fields
left empty in a request are filled with the same automatic choices the
dashboard makes.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class Table(BaseModel):
    columns: list[str]
    rows: list[dict[str, Any]]
    total_rows: int


class Health(BaseModel):
    status: str
    version: str
    datasets: int


class Column(BaseModel):
    column: str
    role: str
    kind: str | None = None
    unique: int
    missing_pct: float
    sample_values: list[Any] = []


class DatasetInfo(BaseModel):
    id: str
    name: str
    sheet: str | None
    sheets: list[str]
    rows: int
    columns: int
    primary_time: str | None
    column_info: list[Column] = []


class QualityIssue(BaseModel):
    severity: str
    message: str
    column: str | None = None


class AskRequest(BaseModel):
    question: str = Field(..., examples=["which doctors are in Pune North territory"])


class AskResponse(BaseModel):
    question: str
    answer: str | None
    notice: str | None
    error: str | None
    table: Table | None
    chart: dict[str, Any] | None


class Finding(BaseModel):
    severity: str
    title: str
    detail: str
    column: str | None = None


class TargetResponse(BaseModel):
    target: str
    kind: str
    positive_label: Any = None
    rows: int
    labelled: int
    positive_rate: float | None
    findings: list[Finding]
    split_method: str | None
    split_summary: str | None
    split: Table | None
    segments: list[dict[str, Any]]
    features: Table
    excluded: list[dict[str, str]]
    markdown: str


class JobStarted(BaseModel):
    job_id: str
    status: str
    status_url: str


class JobStatus(BaseModel):
    job_id: str
    status: Literal["running", "done", "error"]
    messages: list[str]
    error: str | None
    result: dict[str, Any] | None = None


class ModelRequest(BaseModel):
    goal_type: Literal["classify", "rank", "regress"] = "rank"
    target: str | None = Field(None, description="Column to predict; empty = the suggested target")


class ScoreRequest(BaseModel):
    rows: list[dict[str, Any]] = Field(..., description="Rows with the model's feature columns")


class ScoreResponse(BaseModel):
    model: str
    task: str
    predictions: list[float | None]


class RecommendRequest(BaseModel):
    sheet: str | None = Field(None, description="Interaction sheet; empty = detected")
    user: str | None = None
    item: str | None = None
    time: str | None = None
    outcome: str | None = None
    success_values: list[Any] | None = None
    group: str | None = Field(None, description="Planning group label; 'none' = no grouping; empty = detected")
    period: Literal["days", "month"] = "month"
    days: int = 5
    start: str | None = Field(None, description="ISO date; with period=month, that date's month is planned")
    capacity: int | None = None
    min_gap_days: int | None = None
    one_owner: bool = True


class ForecastRequest(BaseModel):
    sheet: str | None = None
    time: str | None = None
    measure: str | None = Field(None, description="Column, or '__rows__' to count rows; empty = detected")
    aggregation: Literal["sum", "mean", "nunique", "count"] | None = None
    group: str | None = None
    freq: Literal["D", "W", "M", "Q"] | None = None
    horizon: int | None = None


class SegmentRequest(BaseModel):
    sheet: str | None = None
    unit: str | None = Field(None, description="Key column for one unit per value; 'rows' = each row; empty = detected")
    features: list[str] | None = None
    method: Literal["kmeans", "bands"] = "kmeans"
    k: int | None = None
    band_measure: str | None = None
    anomaly_share: float = 0.01


class WhyRequest(BaseModel):
    sheet: str | None = None
    time: str | None = None
    measure: str | None = None
    aggregation: Literal["sum", "mean", "count"] | None = None
    freq: Literal["D", "W", "M", "Q"] | None = None
    compare: Literal["previous", "year"] = "previous"
    dimensions: list[str] | None = None
    attention: str | None = Field(None, description="Entity column to rank; 'none' = no ranking; empty = detected")
    higher_is_better: bool = True


class KpiRequest(BaseModel):
    domain: str | None = Field(None, description="Domain file name in the domains folder (e.g. 'pharma_sfa')")
    domain_yaml: str | None = Field(None, description="A domain file's YAML text, instead of a stored file")
    freq: Literal["W", "M", "Q"] = "M"
    group_role: str | None = None


class LlmStatus(BaseModel):
    provider: str | None
    model: str | None
    available: bool
    detail: str


class GoalText(BaseModel):
    text: str = Field(..., examples=["forecast revenue by region for the next 6 months"])


class GoalSpec(BaseModel):
    """A goal as structured fields; empty fields get automatic choices."""

    task: Literal["rank", "classify", "regress", "recommend", "forecast", "segment", "why", "ask"] | None = None
    sheet: str | None = None
    target: str | None = None
    time: str | None = None
    measure: str | None = None
    aggregation: str | None = None
    group: str | None = None
    freq: Literal["D", "W", "M", "Q"] | None = None
    horizon: int | None = None
    compare: Literal["previous", "year"] | None = None
    dimensions: list[str] | None = None
    attention: str | None = None
    user: str | None = None
    item: str | None = None
    outcome: str | None = None
    period: Literal["days", "month"] | None = None
    days: int | None = None
    unit: str | None = None
    features: list[str] | None = None
    question: str | None = None


class GoalQuestion(BaseModel):
    field: str
    text: str
    options: list[Any] = []


class GoalPlanOut(BaseModel):
    spec: GoalSpec
    text: str
    source: str
    title: str
    summary: str
    ready: bool
    tab: str | None
    assumptions: list[str]
    warnings: list[str]
    questions: list[GoalQuestion]
    options: dict[str, list[Any]]
    llm_error: str | None = None


class GoalRun(BaseModel):
    tab: str | None
    job_id: str | None = None
    status_url: str | None = None
    answer: AskResponse | None = None


class DomainInfo(BaseModel):
    name: str
    file: str
    usable: int
    total: int
    error: str = ""
