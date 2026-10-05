"""
Offline evaluation of prioritisation strategies.

Only contacts that actually happened have known outcomes, so the backtest
asks: in each held-out week, if users had prioritised the top share of
that week's contacts by a strategy, how often would those contacts have
succeeded? Strategies:

    Model                     predicted probability (fitted before the test weeks)
    Most overdue first        longest time since the last contact
    Highest past success rate success rate of earlier contacts
    Random                    no prioritisation (averaged over many draws)
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .history import SUCCESS

TOP_SHARE = 0.2
RANDOM_DRAWS = 50
RANDOM_STATE = 42


def backtest(
    rows: pd.DataFrame,
    time_column: str,
    model_scores: pd.Series,
    top_share: float = TOP_SHARE,
) -> pd.DataFrame:

    test = rows.loc[model_scores.index].copy()
    test["_week"] = test[time_column].dt.to_period("W").dt.start_time

    strategies = {
        "Model": model_scores,
        "Most overdue first": test["DaysSinceLastContact"].fillna(10**6),
        "Highest past success rate": test["PriorSuccessRate"].fillna(-1),
    }

    overall = test[SUCCESS].mean()
    total_successes = test[SUCCESS].sum()
    results = []

    for name, scores in strategies.items():
        picked = _top_per_week(test, scores, top_share)
        results.append(_row(name, test.loc[picked, SUCCESS], overall, total_successes))

    rng = np.random.default_rng(RANDOM_STATE)
    draws = []
    for _ in range(RANDOM_DRAWS):
        picked = _top_per_week(test, pd.Series(rng.random(len(test)), index=test.index), top_share)
        draws.append(test.loc[picked, SUCCESS])
    random_rate = float(np.mean([draw.mean() for draw in draws]))
    random_picked = int(np.mean([len(draw) for draw in draws]))
    results.append({
        "Strategy": "Random",
        "Contacts picked": random_picked,
        "Success rate": round(random_rate, 4),
        "Lift": round(random_rate / overall, 2) if overall else np.nan,
        "Successes captured": round(random_rate * random_picked / total_successes, 3) if total_successes else np.nan,
    })

    return pd.DataFrame(results)


def _top_per_week(test: pd.DataFrame, scores: pd.Series, share: float) -> pd.Index:
    picked = []
    for _, week in test.groupby("_week"):
        k = max(1, math.ceil(len(week) * share))
        picked.extend(scores.loc[week.index].sort_values(ascending=False, kind="stable").index[:k])
    return pd.Index(picked)


def _row(name: str, outcomes: pd.Series, overall: float, total_successes: float) -> dict:
    rate = float(outcomes.mean()) if len(outcomes) else float("nan")
    return {
        "Strategy": name,
        "Contacts picked": int(len(outcomes)),
        "Success rate": round(rate, 4),
        "Lift": round(rate / overall, 2) if overall else np.nan,
        "Successes captured": round(outcomes.sum() / total_successes, 3) if total_successes else np.nan,
    }
