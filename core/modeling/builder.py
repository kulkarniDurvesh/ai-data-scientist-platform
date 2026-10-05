"""
Goal -> trained, evaluated, explained model with scores.

    1. prepare features from the GoalSpec
    2. split honestly (time-based when dated)
    3. compare candidates on the validation part
    4. score every candidate once on the test part (for the comparison table)
    5. explain the chosen model; refit it on all labelled rows
    6. score the rows that need a prediction (latest period or unlabelled)
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from core.schema_inference import DatasetSchema, humanize
from core.target_analysis import encode_binary

from . import models
from .explain import ReasonBuilder, importance
from .features import build_preprocessor, prepare_frame, split_columns
from .goal import GoalSpec
from .split import Split, make_split

# Fewer positives than this in training makes results unreliable.
MIN_POSITIVES = 30
MAX_SCORED_ROWS = 5000

Progress = Callable[[str], None]


@dataclass
class ModelResult:
    spec: GoalSpec
    task: str
    split: Split
    candidates: pd.DataFrame
    best_name: str
    test_metrics: dict[str, float]
    baseline_metrics: dict[str, float]
    importance: pd.DataFrame
    scored: pd.DataFrame
    scored_description: str
    lift: pd.DataFrame | None
    pipeline: Pipeline
    numeric: list[str]
    categorical: list[str]
    warnings: list[str] = field(default_factory=list)
    seconds: float = 0.0
    summary: str = ""
    # Predictions of the chosen model on the test rows, made before the
    # final refit (safe to use for backtests).
    test_scores: pd.Series | None = None

    def predict(self, dataframe: pd.DataFrame) -> np.ndarray:
        X = prepare_frame(dataframe, self.numeric, self.categorical)
        if self.task == "classification":
            return self.pipeline.predict_proba(X)[:, 1]
        return self.pipeline.predict(X)


def build_model(
    dataframe: pd.DataFrame,
    schema: DatasetSchema,
    spec: GoalSpec,
    progress: Progress | None = None,
) -> ModelResult:

    started = time.time()
    say = progress or (lambda message: None)
    task = spec.task
    notes: list[str] = []

    say("Preparing features")
    numeric, categorical = split_columns(schema, spec.features)
    X_all = prepare_frame(dataframe, numeric, categorical)

    if task == "classification":
        y_all, positive = encode_binary(dataframe[spec.target])
        spec.positive_label = positive
    else:
        y_all = pd.to_numeric(dataframe[spec.target], errors="coerce")

    split = make_split(dataframe, y_all, spec.time_column, stratify=task == "classification")

    if len(split.test) == 0 or len(split.validation) == 0:
        raise ValueError("Not enough labelled rows to train and test a model.")

    if task == "classification":
        positives = int(y_all.loc[split.train].sum())
        if positives < MIN_POSITIVES:
            notes.append(
                f"Only {positives} positive examples in training: results may not hold on new data."
            )
        if y_all.loc[split.test].nunique() < 2:
            notes.append("The test period contains only one class, so ranking metrics can't be computed there.")

    def fit(name: str, rows: pd.Index) -> Pipeline:
        estimator, scaled = models.candidates(task)[name]
        pipeline = Pipeline([
            ("prepare", build_preprocessor(numeric, categorical, scale=scaled)),
            ("model", clone(estimator)),
        ])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipeline.fit(X_all.loc[rows], y_all.loc[rows].to_numpy())
        return pipeline

    def predict(pipeline: Pipeline, rows: pd.Index) -> np.ndarray:
        X = X_all.loc[rows]
        if task == "classification":
            return pipeline.predict_proba(X)[:, 1]
        return pipeline.predict(X)

    # -- 3. compare on validation ------------------------------------------
    rows = []
    fitted_for_test: dict[str, Pipeline] = {}

    for name in models.candidates(task):
        say(f"Training {name.lower()}")
        selector = fit(name, split.inner_train)
        valid_score = models.selection_score(
            task, y_all.loc[split.validation].to_numpy(), predict(selector, split.validation),
        )

        # -- 4. one test evaluation per candidate --------------------------
        model = fit(name, split.train)
        fitted_for_test[name] = model
        test_prediction = predict(model, split.test)
        y_test = y_all.loc[split.test].to_numpy()

        metrics = (
            models.classification_metrics(y_test, test_prediction)
            if task == "classification"
            else models.regression_metrics(y_test, test_prediction)
        )
        rows.append({"Model": name, "validation": valid_score, **metrics})

    table = pd.DataFrame(rows)
    real = table[table["Model"] != models.BASELINE]
    best_name = (
        real.loc[real["validation"].idxmax(), "Model"]
        if real["validation"].notna().any()
        else real.iloc[0]["Model"]
    )

    best_metrics = table.set_index("Model").loc[best_name].to_dict()
    baseline_metrics = table.set_index("Model").loc[models.BASELINE].to_dict()

    # -- 5. explain and refit ------------------------------------------------
    say("Explaining the chosen model")
    best_test_model = fitted_for_test[best_name]
    scoring = "average_precision" if task == "classification" else "neg_mean_absolute_error"
    if task == "classification" and y_all.loc[split.test].nunique() < 2:
        scoring = "neg_brier_score"

    importances = importance(
        best_test_model, X_all.loc[split.test], y_all.loc[split.test].to_numpy(), scoring,
    )

    test_scores = pd.Series(predict(best_test_model, split.test), index=split.test)

    lift = None
    if task == "classification" and y_all.loc[split.test].nunique() == 2:
        lift = models.lift_table(y_all.loc[split.test].to_numpy(), test_scores.to_numpy())

    say("Refitting on all labelled rows")
    labelled = y_all.index[y_all.notna()]
    final = fit(best_name, labelled)

    # -- 6. score ----------------------------------------------------------
    say("Scoring")
    score_rows, scored_description = _rows_to_score(dataframe, spec, y_all)
    score_rows = score_rows[:MAX_SCORED_ROWS]
    scores = predict(final, score_rows)

    reasons = ReasonBuilder(
        X_all.loc[labelled], predict(final, labelled), importances, numeric, categorical,
    ).explain(X_all.loc[score_rows], scores)

    scored = _scored_table(dataframe, spec, score_rows, scores, reasons, y_all, task)

    result = ModelResult(
        spec=spec,
        task=task,
        split=split,
        candidates=_candidate_table(table, task, best_name),
        best_name=best_name,
        test_metrics=best_metrics,
        baseline_metrics=baseline_metrics,
        importance=importances,
        scored=scored,
        scored_description=scored_description,
        lift=lift,
        pipeline=final,
        numeric=numeric,
        categorical=categorical,
        warnings=notes,
        seconds=round(time.time() - started, 1),
        test_scores=test_scores,
    )
    result.summary = summarize(result)

    if task == "classification" and not best_metrics.get("pr_auc", 0) > baseline_metrics.get("pr_auc", 0):
        result.warnings.append("The chosen model does not beat the baseline on the test period.")
    if task == "regression" and not best_metrics["mae"] < baseline_metrics["mae"]:
        result.warnings.append("The chosen model does not beat the baseline on the test period.")

    say("Done")
    return result


def _rows_to_score(
    dataframe: pd.DataFrame,
    spec: GoalSpec,
    y: pd.Series,
) -> tuple[pd.Index, str]:
    """Rows a prediction is useful for: unlabelled rows, else the latest period."""

    unlabelled = y.index[y.isna()]
    if len(unlabelled):
        return unlabelled, f"{len(unlabelled):,} rows without a known {humanize(spec.target)}"

    if spec.time_column:
        latest = dataframe[spec.time_column].max()
        rows = dataframe.index[dataframe[spec.time_column] == latest]
        label = pd.Timestamp(latest).strftime("%Y-%m-%d") if not pd.isna(latest) else str(latest)
        return rows, f"the latest period ({humanize(spec.time_column)} = {label}), {len(rows):,} rows"

    return dataframe.index, f"all {len(dataframe):,} rows"


def _scored_table(
    dataframe: pd.DataFrame,
    spec: GoalSpec,
    rows: pd.Index,
    scores: np.ndarray,
    reasons: list[str],
    y: pd.Series,
    task: str,
) -> pd.DataFrame:

    table = pd.DataFrame(index=rows)

    if spec.entity and spec.entity in dataframe.columns:
        table[spec.entity] = dataframe.loc[rows, spec.entity]

    score_column = "Probability" if task == "classification" else f"Predicted {spec.target}"
    table[score_column] = np.round(scores, 4)
    table["Reasons"] = reasons

    known = y.loc[rows]
    if known.notna().any():
        table[f"Actual {spec.target}"] = dataframe.loc[rows, spec.target]

    table = table.sort_values(score_column, ascending=False)
    table.insert(0, "Rank", np.arange(1, len(table) + 1))
    return table.reset_index(drop=True)


def _candidate_table(table: pd.DataFrame, task: str, best_name: str) -> pd.DataFrame:
    if task == "classification":
        columns = {
            "Model": "Model",
            "validation": "Validation PR-AUC",
            "pr_auc": "Test PR-AUC",
            "roc_auc": "Test ROC-AUC",
            "recall_at_top": "Top 20% capture",
            "lift_at_top": "Top 20% lift",
        }
    else:
        columns = {
            "Model": "Model",
            "validation": "Validation MAE",
            "mae": "Test MAE",
            "rmse": "Test RMSE",
            "r2": "Test R²",
        }

    out = table[list(columns)].rename(columns=columns).copy()

    if task == "regression":
        out["Validation MAE"] = -out["Validation MAE"]

    for column in out.columns[1:]:
        out[column] = out[column].astype(float).round(3)

    out.insert(1, "Chosen", np.where(out["Model"] == best_name, "✔", ""))
    return out


def summarize(result: ModelResult) -> str:
    best, base = result.test_metrics, result.baseline_metrics
    name = result.best_name

    if result.task == "classification":
        if np.isnan(best.get("recall_at_top", np.nan)):
            return f"{name} was chosen on the validation window."
        share = int(best["top_share"] * 100)
        return (
            f"{name} was chosen. On the test period, the top {share}% of its ranked "
            f"list captures {best['recall_at_top']:.0%} of all positives "
            f"({best['lift_at_top']:.1f}× the average rate), versus {share}% "
            f"for a random list. PR-AUC {best['pr_auc']:.3f} vs {base['pr_auc']:.3f} "
            f"for the baseline (ROC-AUC {best['roc_auc']:.3f})."
        )

    improvement = (1 - best["mae"] / base["mae"]) if base["mae"] else float("nan")
    return (
        f"{name} was chosen. On the test rows its average error is "
        f"{best['mae']:.3g} (baseline {base['mae']:.3g}, {improvement:.0%} better); "
        f"R² {best['r2']:.3f}."
    )


def result_summary_dict(result: ModelResult) -> dict[str, Any]:
    """Plain values for saving alongside the model."""

    return {
        "goal": result.spec.to_dict(),
        "task": result.task,
        "best_model": result.best_name,
        "test_metrics": {k: float(v) for k, v in result.test_metrics.items() if isinstance(v, (int, float))},
        "baseline_metrics": {k: float(v) for k, v in result.baseline_metrics.items() if isinstance(v, (int, float))},
        "split": result.split.description,
        "summary": result.summary,
        "warnings": result.warnings,
        "numeric": result.numeric,
        "categorical": result.categorical,
    }
