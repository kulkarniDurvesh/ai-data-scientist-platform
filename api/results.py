"""JSON views of the platform's result objects (models, plans, forecasts, segments, investigations)."""

from __future__ import annotations

from typing import Any

from core.agents import AgentRun
from core.forecast import ForecastResult
from core.modeling import ModelResult
from core.recommend import RecommendResult
from core.segment import SegmentResult
from core.why import Investigation

from .convert import plain, table

ROWS = 500


def model_result(result: ModelResult, rows: int = ROWS) -> dict[str, Any]:
    return {
        "type": "model",
        "goal": result.spec.to_dict(),
        "task": result.task,
        "best_model": result.best_name,
        "summary": result.summary,
        "warnings": result.warnings,
        "split": result.split.description,
        "test_metrics": plain(result.test_metrics),
        "baseline_metrics": plain(result.baseline_metrics),
        "candidates": table(result.candidates),
        "importance": table(result.importance),
        "lift": table(result.lift),
        "scored_description": result.scored_description,
        "scored": table(result.scored, limit=rows),
        "seconds": result.seconds,
    }


def recommend_result(result: RecommendResult, rows: int = ROWS, user: str | None = None) -> dict[str, Any]:
    plan = result.plan
    if user is not None and not plan.empty:
        plan = plan[plan[result.roles.user].astype(str) == str(user)]
    settings = result.settings
    return {
        "type": "recommendation",
        "roles": plain(result.roles.to_dict()),
        "settings": {
            "start": plain(settings.start), "days": settings.days, "capacity": settings.capacity,
            "min_gap_days": settings.min_gap_days, "period": settings.period, "one_owner": settings.one_owner,
        },
        "summary": result.summary,
        "warnings": result.warnings,
        "success_model": result.model.best_name,
        "success_model_metrics": plain(result.model.test_metrics),
        "backtest": table(result.backtest),
        "by_group": table(result.summary_table),
        "by_user": table(result.user_table),
        "plan": table(plan, limit=rows),
        "seconds": result.seconds,
    }


def forecast_result(result: ForecastResult, rows: int = ROWS) -> dict[str, Any]:
    spec = result.spec
    return {
        "type": "forecast",
        "spec": {"time": spec.time, "measure": spec.measure, "aggregation": spec.aggregation,
                 "group": spec.group, "freq": spec.freq, "horizon": spec.horizon},
        "measure_label": result.measure_label(),
        "best_model": result.best_model,
        "summary": result.summary,
        "notes": result.notes,
        "diagnostics": plain(result.diagnostics),
        "metrics": table(result.metrics),
        "forecasts": table(result.forecasts, limit=rows),
        "history": {name: {plain(k): plain(v) for k, v in series.items()} for name, series in result.series.items()},
        "seconds": result.seconds,
    }


def segment_result(result: SegmentResult, rows: int = ROWS) -> dict[str, Any]:
    clustering = result.clustering
    correlation = result.correlation
    return {
        "type": "segments",
        "unit": result.units.unit,
        "features": result.units.features,
        "method": clustering.method,
        "segments": clustering.k,
        "silhouette": plain(clustering.silhouette),
        "summary": result.summary,
        "notes": result.notes,
        "profiles": table(clustering.profiles),
        "k_scores": table(clustering.k_scores),
        "anomalies": table(result.anomalies, limit=rows),
        "units": table(result.table, limit=rows),
        "correlation": {
            "redundant": table(correlation.redundant),
            "vif": table(correlation.vif),
            "categorical": table(correlation.categorical),
            "top_pairs": table(correlation.pairs),
        },
        "seconds": result.seconds,
    }


def investigation_result(result: Investigation) -> dict[str, Any]:
    comparison = result.comparison
    return {
        "type": "investigation",
        "measure_label": result.label(),
        "current_period": plain(comparison.current_period),
        "previous_period": plain(comparison.previous_period),
        "current": plain(comparison.current),
        "previous": plain(comparison.previous),
        "change": plain(comparison.change),
        "change_pct": plain(comparison.change_pct),
        "explanation": [
            {"text": step.text, "path": [{"dimension": d, "value": plain(v)} for d, v in step.path],
             "dimension": step.breakdown.dimension if step.breakdown else None}
            for step in result.steps
        ],
        "breakdowns": [
            {"dimension": item.dimension, "concentration": plain(item.concentration), "table": table(item.table, limit=50)}
            for item in result.breakdowns
        ],
        "trend": {plain(k): plain(v) for k, v in result.trend.items()},
        "attention_dimension": result.attention_dimension,
        "attention": table(result.attention),
        "unusual": table(result.unusual),
        "notes": result.notes,
    }


def any_result(result: Any) -> dict[str, Any] | None:
    if isinstance(result, AgentRun):
        return {"type": "agent", **plain(result.to_dict())}
    for kind, view in (
        (ModelResult, model_result),
        (RecommendResult, recommend_result),
        (ForecastResult, forecast_result),
        (SegmentResult, segment_result),
        (Investigation, investigation_result),
    ):
        if isinstance(result, kind):
            return view(result)
    return None
