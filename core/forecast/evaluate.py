"""
Rolling-origin backtest and forecast metrics.

Several cut-off points near the end of the history; at each, every model
is fitted on the data up to the cut-off and forecasts the next h periods,
which are compared with what actually happened. No random splits: the
future is never used to predict the past.

    MAE    mean absolute error
    sMAPE  symmetric mean absolute percentage error
    MASE   MAE divided by the in-sample seasonal-naive MAE (< 1 beats it)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import models

MAX_ORIGINS = 4


def origins(length: int, h: int, m: int) -> list[int]:
    """Cut-off positions (history length used for fitting) for the backtest."""

    minimum = max(m + 2, 6) if length >= 2 * m + h else max(4, length // 2)
    available = list(range(length - h, minimum - 1, -max(1, h // 2 or 1)))
    return sorted(available[:MAX_ORIGINS])


def smape(actual: np.ndarray, forecast: np.ndarray) -> float:
    denominator = np.abs(actual) + np.abs(forecast)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(denominator == 0, 0.0, 2 * np.abs(actual - forecast) / denominator)
    return float(np.mean(terms))


def scale(y: np.ndarray, m: int) -> float:
    """In-sample MAE of the seasonal naive (or naive) forecast."""

    lag = m if len(y) > m else 1
    differences = np.abs(y[lag:] - y[:-lag])
    value = float(np.mean(differences)) if len(differences) else 0.0
    return value if value > 0 else max(float(np.mean(np.abs(y))), 1e-9)


def backtest(
    series: dict[str, pd.Series],
    h: int,
    m: int,
    freq: str,
    model_names: list[str],
    use_global: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (errors, metrics):
        errors   one row per series / model / origin / step with the error
        metrics  per model: MAE, sMAPE, MASE averaged over series and origins
    """

    length = min(len(values) for values in series.values())
    cut_offs = origins(length, h, m)
    if not cut_offs:
        raise ValueError("Not enough history to backtest this horizon: shorten it or use a finer period.")

    rows = []
    for cut in cut_offs:
        histories = {name: values.iloc[:cut] for name, values in series.items()}
        futures = {name: values.iloc[cut:cut + h] for name, values in series.items()}

        predictions: dict[str, dict[str, np.ndarray]] = {}
        for model in model_names:
            predictions[model] = {
                name: models.local_forecast(model, history.to_numpy(), len(futures[name]), m)
                for name, history in histories.items()
            }
        if use_global:
            steps = min(len(future) for future in futures.values())
            predictions[models.GLOBAL_GB] = models.global_forecast(histories, steps, m, freq)

        for model, by_series in predictions.items():
            for name, forecast in by_series.items():
                actual = futures[name].to_numpy()[: len(forecast)]
                y_scale = scale(histories[name].to_numpy(), m)
                for step, (a, f) in enumerate(zip(actual, forecast), start=1):
                    rows.append({
                        "series": name, "model": model, "origin": cut, "step": step,
                        "actual": a, "forecast": f, "error": a - f, "scale": y_scale,
                    })

    errors = pd.DataFrame(rows)

    def summarize(group: pd.DataFrame) -> pd.Series:
        return pd.Series({
            "MAE": float(np.mean(np.abs(group["error"]))),
            "sMAPE": smape(group["actual"].to_numpy(), group["forecast"].to_numpy()),
            "MASE": float(np.mean(np.abs(group["error"]) / group["scale"])),
        })

    metrics = errors.groupby("model").apply(summarize, include_groups=False).reset_index()
    return errors, metrics.sort_values("MASE").reset_index(drop=True)


def interval_widths(errors: pd.DataFrame, model: str, series_name: str, h: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Approximate 80% / 95% half-widths per forecast step from the model's
    backtest errors on this series (pooled and widened with the step when
    there are too few errors at a step).
    """

    own = errors[(errors["model"] == model) & (errors["series"] == series_name)]
    overall = float(np.std(own["error"])) if len(own) > 1 else 0.0

    sigma = []
    for step in range(1, h + 1):
        at_step = own.loc[own["step"] == step, "error"]
        if len(at_step) >= 3:
            sigma.append(float(np.std(at_step)))
        else:
            sigma.append(overall * np.sqrt(step))

    sigma = np.maximum.accumulate(np.array(sigma))
    return 1.2816 * sigma, 1.96 * sigma
