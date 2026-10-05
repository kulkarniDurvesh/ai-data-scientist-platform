"""
Forecast pipeline.

    table + SeriesSpec
        -> regular series (total and per group)
        -> diagnostics (trend / seasonality strength)
        -> rolling-origin backtest of every model
        -> best model by MASE, refit on all history
        -> forecasts with approximate intervals
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from core.schema_inference import humanize

from . import models
from .evaluate import backtest, interval_widths
from .series import ROW_COUNT, TOTAL, SeriesSet, SeriesSpec, build_series, future_index

Progress = Callable[[str], None]


@dataclass
class ForecastResult:
    spec: SeriesSpec
    series: dict[str, pd.Series]
    best_model: str
    metrics: pd.DataFrame
    forecasts: pd.DataFrame          # series, period, forecast, lower/upper 80 and 95
    diagnostics: dict[str, float]
    decomposition: pd.DataFrame | None
    notes: list[str] = field(default_factory=list)
    summary: str = ""
    seconds: float = 0.0

    def measure_label(self) -> str:
        if self.spec.measure == ROW_COUNT:
            return "Number of rows"
        prefix = {"sum": "Total", "mean": "Average", "nunique": "Distinct"}.get(self.spec.aggregation, "")
        name = humanize(self.spec.measure)
        if prefix and name.lower().startswith(prefix.lower()):
            return name
        return f"{prefix} {name}".strip()


def build_forecast(
    dataframe: pd.DataFrame,
    spec: SeriesSpec,
    progress: Progress | None = None,
) -> ForecastResult:

    started = time.time()
    say = progress or (lambda message: None)

    say("Building time series")
    series_set = build_series(dataframe, spec)
    series = series_set.series
    m, h = spec.season, spec.horizon
    length = len(series[TOTAL])

    if h >= length // 2:
        raise ValueError(f"A horizon of {h} is too long for {length} periods of history.")

    say("Measuring trend and seasonality")
    diagnostics, decomposition = _diagnostics(series[TOTAL], m)

    candidates = [name for name in models.LOCAL_MODELS if name != models.SEASONAL_NAIVE or length >= m + 2]
    use_global = len(series) * length >= 60 and length >= 10

    say("Backtesting models on past periods")
    errors, metrics = backtest(series, h, m, spec.freq, candidates, use_global)

    real = metrics[~metrics["model"].isin(models.BASELINES)]
    best = real.iloc[0]["model"] if len(real) else metrics.iloc[0]["model"]

    say(f"Forecasting with {best}")
    forecasts = _final_forecasts(series, best, h, m, spec.freq, errors)

    result = ForecastResult(
        spec=spec,
        series=series,
        best_model=best,
        metrics=_metric_table(metrics, best),
        forecasts=forecasts,
        diagnostics=diagnostics,
        decomposition=decomposition,
        notes=list(series_set.notes),
        seconds=round(time.time() - started, 1),
    )
    result.summary = _summarize(result)

    baseline = metrics[metrics["model"].isin(models.BASELINES)]["MASE"].min()
    if float(metrics.loc[metrics["model"] == best, "MASE"].iloc[0]) >= baseline:
        result.notes.append("No model beats the simple baselines here; treat the forecast with caution.")

    say("Done")
    return result


def _final_forecasts(
    series: dict[str, pd.Series],
    model: str,
    h: int,
    m: int,
    freq: str,
    errors: pd.DataFrame,
) -> pd.DataFrame:

    if model == models.GLOBAL_GB:
        values = models.global_forecast(series, h, m, freq)
    else:
        values = {name: models.local_forecast(model, s.to_numpy(), h, m) for name, s in series.items()}

    rows = []
    for name, forecast in values.items():
        index = future_index(series[name].index[-1], freq, h)
        w80, w95 = interval_widths(errors, model, name, h)
        floor = 0.0 if (series[name] >= 0).all() else -np.inf
        for step, (period, value) in enumerate(zip(index, forecast)):
            rows.append({
                "Series": name,
                "Period": period,
                "Forecast": max(float(value), floor),
                "Low 80%": max(float(value - w80[step]), floor),
                "High 80%": float(value + w80[step]),
                "Low 95%": max(float(value - w95[step]), floor),
                "High 95%": float(value + w95[step]),
            })

    return pd.DataFrame(rows)


def _diagnostics(y: pd.Series, m: int) -> tuple[dict[str, float], pd.DataFrame | None]:
    """Trend and seasonal strength (0..1) from an STL decomposition."""

    # Two cycles let the seasonal component absorb almost anything, so
    # strengths are only reported with three or more full seasons.
    if len(y) < 3 * m or m < 2:
        slope = np.polyfit(np.arange(len(y)), y.to_numpy(), 1)[0] if len(y) > 2 else 0.0
        return {"trend_strength": float("nan"), "seasonal_strength": float("nan"),
                "slope_per_period": float(slope)}, None

    from statsmodels.tsa.seasonal import STL

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = STL(y.to_numpy(), period=m, robust=True).fit()

    resid, trend, seasonal = fit.resid, fit.trend, fit.seasonal
    strength = lambda part: float(max(0.0, 1 - np.var(resid) / np.var(part + resid))) if np.var(part + resid) else 0.0

    decomposition = pd.DataFrame(
        {"Observed": y.to_numpy(), "Trend": trend, "Seasonal": seasonal, "Remainder": resid},
        index=y.index,
    )
    slope = float(np.polyfit(np.arange(len(trend)), trend, 1)[0])

    return {"trend_strength": strength(trend), "seasonal_strength": strength(seasonal),
            "slope_per_period": slope}, decomposition


def _metric_table(metrics: pd.DataFrame, best: str) -> pd.DataFrame:
    table = metrics.rename(columns={"model": "Model"}).copy()
    table.insert(1, "Chosen", np.where(table["Model"] == best, "✔", ""))
    table.insert(2, "Type", np.where(table["Model"].isin(models.BASELINES), "baseline", "model"))
    for column in ("MAE", "sMAPE", "MASE"):
        table[column] = table[column].astype(float).round(3)
    return table


def _summarize(result: ForecastResult) -> str:
    table = result.metrics.set_index("Model")
    best = table.loc[result.best_model]
    baselines = table[table["Type"] == "baseline"]
    reference = baselines["MASE"].idxmin() if len(baselines) else None

    total = result.forecasts[result.forecasts["Series"] == TOTAL]
    first, last = total["Period"].min(), total["Period"].max()

    text = (
        f"{result.best_model} was chosen on the rolling backtest "
        f"(MASE {best['MASE']:.2f}, sMAPE {best['sMAPE']:.1%}"
    )
    if reference:
        text += f"; best baseline {reference.lower()} MASE {baselines.loc[reference, 'MASE']:.2f}"
    text += (
        f"). {result.measure_label()} forecast for {first:%Y-%m-%d} to {last:%Y-%m-%d}: "
        f"{total['Forecast'].sum():,.0f} in total over {len(total)} period(s)."
    )

    trend, seasonal = result.diagnostics["trend_strength"], result.diagnostics["seasonal_strength"]
    if not np.isnan(trend):
        text += f" Trend strength {trend:.2f}, seasonality strength {seasonal:.2f} (0 = none, 1 = strong)."
    else:
        text += (
            f" Trend and seasonality strength need at least three full seasons "
            f"({3 * result.spec.season} periods); the history has {len(result.series[TOTAL])}."
        )
    return text
