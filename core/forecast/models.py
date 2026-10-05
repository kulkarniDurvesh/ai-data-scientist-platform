"""
Forecasting models behind one interface: history in, next h values out.

    Naive               last value repeated
    Seasonal naive      value from the same period one season earlier
    Drift               straight line from the first to the last value
    ETS                 exponential smoothing (damped trend, seasonality if 2+ seasons)
    Theta               Theta method (strong benchmark in forecasting competitions)
    ARIMA               ARIMA(1,1,1)
    Gradient boosting   one model for all series on lag and calendar features
"""

from __future__ import annotations

import warnings
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

NAIVE = "Naive (last value)"
SEASONAL_NAIVE = "Seasonal naive"
BASELINES = {NAIVE, SEASONAL_NAIVE}
GLOBAL_GB = "Gradient boosting (lags, all series)"


def naive(y: np.ndarray, h: int, m: int) -> np.ndarray:
    return np.repeat(y[-1], h)


def seasonal_naive(y: np.ndarray, h: int, m: int) -> np.ndarray:
    if len(y) < m:
        return naive(y, h, m)
    last_season = y[-m:]
    return np.array([last_season[i % m] for i in range(h)])


def drift(y: np.ndarray, h: int, m: int) -> np.ndarray:
    if len(y) < 2:
        return naive(y, h, m)
    slope = (y[-1] - y[0]) / (len(y) - 1)
    return y[-1] + slope * np.arange(1, h + 1)


def ets(y: np.ndarray, h: int, m: int) -> np.ndarray:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    seasonal = "add" if len(y) >= 2 * m and m > 1 else None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ExponentialSmoothing(
            y, trend="add", damped_trend=True, seasonal=seasonal,
            seasonal_periods=m if seasonal else None, initialization_method="estimated",
        ).fit()
    return np.asarray(model.forecast(h))


def theta(y: np.ndarray, h: int, m: int) -> np.ndarray:
    from statsmodels.tsa.forecasting.theta import ThetaModel

    deseasonalize = len(y) >= 2 * m and m > 1 and np.all(y > 0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ThetaModel(y, period=m if deseasonalize else None, deseasonalize=deseasonalize).fit()
    return np.asarray(model.forecast(h))


def arima(y: np.ndarray, h: int, m: int) -> np.ndarray:
    from statsmodels.tsa.arima.model import ARIMA

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ARIMA(y, order=(1, 1, 1)).fit()
    return np.asarray(model.forecast(h))


LOCAL_MODELS: dict[str, Callable[[np.ndarray, int, int], np.ndarray]] = {
    NAIVE: naive,
    SEASONAL_NAIVE: seasonal_naive,
    "Drift": drift,
    "ETS (exponential smoothing)": ets,
    "Theta": theta,
    "ARIMA(1,1,1)": arima,
}


def local_forecast(name: str, y: np.ndarray, h: int, m: int) -> np.ndarray:
    """Forecast with a single-series model, falling back to naive on failure."""

    try:
        values = LOCAL_MODELS[name](y, h, m)
        if not np.all(np.isfinite(values)):
            raise ValueError("non-finite forecast")
        return values
    except Exception:  # noqa: BLE001 - statsmodels can fail on short/flat series
        return naive(y, h, m)


# ----------------------------------------------------------------------
# Global gradient boosting on lag features
# ----------------------------------------------------------------------

def _lags(m: int, length: int) -> list[int]:
    candidates = [1, 2, 3, m] if m > 3 else [1, 2, 3]
    return sorted({lag for lag in candidates if lag < length - 2})


def global_forecast(
    histories: dict[str, pd.Series],
    h: int,
    m: int,
    freq: str,
) -> dict[str, np.ndarray]:
    """One model across all series; multi-step forecasts made recursively."""

    length = min(len(series) for series in histories.values())
    lags = _lags(m, length)
    if not lags:
        return {name: naive(series.to_numpy(), h, m) for name, series in histories.items()}

    scale = {name: max(float(np.mean(np.abs(series))), 1e-9) for name, series in histories.items()}

    def features(values: np.ndarray, position: int, stamp: pd.Timestamp) -> list[float]:
        row = [values[position - lag] for lag in lags]
        row.append(np.mean(values[max(0, position - 3):position]))
        row.append(_season_position(stamp, freq))
        return row

    X, y = [], []
    for name, series in histories.items():
        values = series.to_numpy() / scale[name]
        for position in range(max(lags), len(values)):
            X.append(features(values, position, series.index[position]))
            y.append(values[position])

    if len(y) < 20:
        return {name: naive(series.to_numpy(), h, m) for name, series in histories.items()}

    model = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, random_state=42)
    model.fit(np.array(X), np.array(y))

    forecasts = {}
    for name, series in histories.items():
        values = list(series.to_numpy() / scale[name])
        stamps = list(series.index)
        step = stamps[-1] - stamps[-2] if len(stamps) > 1 else pd.Timedelta(days=30)
        out = []
        for _ in range(h):
            stamp = stamps[-1] + step
            prediction = float(model.predict(np.array([features(np.array(values + [0.0]), len(values), stamp)]))[0])
            values.append(prediction)
            stamps.append(stamp)
            out.append(prediction * scale[name])
        forecasts[name] = np.array(out)

    return forecasts


def _season_position(stamp: pd.Timestamp, freq: str) -> float:
    if freq == "D":
        return float(stamp.dayofweek)
    if freq == "W":
        return float(stamp.isocalendar().week)
    if freq == "Q":
        return float(stamp.quarter)
    return float(stamp.month)
