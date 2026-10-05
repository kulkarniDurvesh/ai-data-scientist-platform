"""
Candidate models and evaluation metrics.

Every goal compares a baseline that uses no features with a linear model
and two tree ensembles. Metrics are chosen for the question being asked:
ranking quality and capture at the top of the list for yes/no targets,
absolute error for numbers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

RANDOM_STATE = 42

# Share of rows treated as "the top of the list" for ranking metrics.
TOP_SHARE = 0.2

BASELINE = "Baseline (no features)"


def candidates(task: str) -> dict[str, tuple[object, bool]]:
    """name -> (estimator, needs scaled inputs)."""

    if task == "classification":
        return {
            BASELINE: (DummyClassifier(strategy="prior"), False),
            "Logistic regression": (
                LogisticRegression(max_iter=2000, class_weight="balanced"),
                True,
            ),
            "Random forest": (
                RandomForestClassifier(
                    n_estimators=300,
                    min_samples_leaf=5,
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                ),
                False,
            ),
            "Gradient boosting": (
                HistGradientBoostingClassifier(
                    learning_rate=0.06,
                    max_iter=300,
                    class_weight="balanced",
                    early_stopping=True,
                    random_state=RANDOM_STATE,
                ),
                False,
            ),
        }

    return {
        BASELINE: (DummyRegressor(strategy="mean"), False),
        "Ridge regression": (Ridge(alpha=1.0), True),
        "Random forest": (
            RandomForestRegressor(
                n_estimators=300,
                min_samples_leaf=5,
                n_jobs=-1,
                random_state=RANDOM_STATE,
            ),
            False,
        ),
        "Gradient boosting": (
            HistGradientBoostingRegressor(
                learning_rate=0.06,
                max_iter=300,
                early_stopping=True,
                random_state=RANDOM_STATE,
            ),
            False,
        ),
    }


def selection_score(task: str, y_true: np.ndarray, prediction: np.ndarray) -> float:
    """Higher is better: PR-AUC for yes/no targets, negative MAE for numbers."""

    if task == "classification":
        if len(np.unique(y_true)) < 2:
            return float("nan")
        return float(average_precision_score(y_true, prediction))
    return -float(mean_absolute_error(y_true, prediction))


def classification_metrics(y_true: np.ndarray, score: np.ndarray) -> dict[str, float]:
    metrics = {
        "positive_rate": float(np.mean(y_true)),
        "brier": float(brier_score_loss(y_true, score)),
    }

    if len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, score))
        metrics["pr_auc"] = float(average_precision_score(y_true, score))
    else:
        metrics["roc_auc"] = float("nan")
        metrics["pr_auc"] = float("nan")

    top = top_capture(y_true, score, TOP_SHARE)
    metrics.update(top)
    return metrics


def top_capture(y_true: np.ndarray, score: np.ndarray, share: float) -> dict[str, float]:
    """How much of the outcome the top `share` of the ranked list captures."""

    n = len(y_true)
    k = max(1, int(round(n * share)))
    order = np.argsort(-score, kind="stable")
    top = np.asarray(y_true)[order[:k]]
    positives = float(np.sum(y_true))
    rate = float(np.mean(y_true)) if n else 0.0

    precision = float(np.mean(top)) if k else 0.0
    recall = float(np.sum(top) / positives) if positives else float("nan")
    lift = precision / rate if rate else float("nan")

    return {"top_share": share, "precision_at_top": precision, "recall_at_top": recall, "lift_at_top": lift}


def regression_metrics(y_true: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, prediction)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, prediction))),
        "r2": float(r2_score(y_true, prediction)) if len(y_true) > 1 else float("nan"),
    }


def lift_table(y_true: np.ndarray, score: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Rows ranked by score, cut into equal groups: capture and lift per group."""

    frame = pd.DataFrame({"y": np.asarray(y_true, dtype=float), "score": score})
    frame = frame.sort_values("score", ascending=False, kind="stable").reset_index(drop=True)
    frame["group"] = (np.arange(len(frame)) * bins // max(len(frame), 1)) + 1

    overall = frame["y"].mean()
    total = frame["y"].sum()

    table = frame.groupby("group").agg(Rows=("y", "size"), Positives=("y", "sum"), Rate=("y", "mean"))
    table["Positives"] = table["Positives"].astype(int)
    table["Lift"] = (table["Rate"] / overall).round(2) if overall else np.nan
    table["Cumulative capture"] = (table["Positives"].cumsum() / total).round(3) if total else np.nan
    table["Top % of list"] = (table["Rows"].cumsum() / len(frame) * 100).round(0).astype(int)
    table["Rate"] = table["Rate"].round(4)

    return table.reset_index(drop=True)[["Top % of list", "Rows", "Positives", "Rate", "Lift", "Cumulative capture"]]
