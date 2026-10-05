"""Forecasting: series building, model backtests and forecasts with intervals (Phase 3c)."""

from .engine import ForecastResult, build_forecast
from .series import (
    DEFAULT_HORIZON,
    ROW_COUNT,
    TOTAL,
    SeriesSpec,
    build_series,
    date_options,
    default_aggregation,
    group_options,
    measure_options,
    suggest_freq,
)

__all__ = [
    "DEFAULT_HORIZON",
    "ROW_COUNT",
    "TOTAL",
    "ForecastResult",
    "SeriesSpec",
    "build_forecast",
    "build_series",
    "date_options",
    "default_aggregation",
    "group_options",
    "measure_options",
    "suggest_freq",
]
