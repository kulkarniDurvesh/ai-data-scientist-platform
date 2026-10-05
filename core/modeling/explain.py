"""
Explanations that work for any model.

    global  -> permutation importance on the test rows: how much the score
               drops when one column is shuffled
    per row -> the most important columns where this row is unusual, in the
               direction that moves its prediction ("high OrderRate (top 10%)")
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from core.schema_inference import format_number, humanize

RANDOM_STATE = 42
MAX_IMPORTANCE_ROWS = 3000
REASON_FEATURES = 8      # most important columns considered for reasons
REASONS_PER_ROW = 3
HIGH, LOW = 0.8, 0.2     # percentile bands that count as notable


def importance(model, X: pd.DataFrame, y: np.ndarray, scoring: str) -> pd.DataFrame:
    if len(X) > MAX_IMPORTANCE_ROWS:
        sample = np.random.default_rng(RANDOM_STATE).choice(len(X), MAX_IMPORTANCE_ROWS, replace=False)
        X, y = X.iloc[sample], y[sample]

    result = permutation_importance(
        model, X, y, scoring=scoring, n_repeats=5,
        random_state=RANDOM_STATE, n_jobs=1,
    )

    table = pd.DataFrame({
        "Feature": X.columns,
        "Importance": result.importances_mean,
        "Spread": result.importances_std,
    })
    return table.sort_values("Importance", ascending=False).reset_index(drop=True)


class ReasonBuilder:
    """
    Learns, from reference rows and their predictions, which way each
    important column pushes the prediction, then explains single rows.
    """

    def __init__(
        self,
        reference: pd.DataFrame,
        prediction: np.ndarray,
        importances: pd.DataFrame,
        numeric: list[str],
        categorical: list[str],
    ):
        positive = importances[importances["Importance"] > 0]
        self.features = list(positive["Feature"].head(REASON_FEATURES))
        self.numeric = set(numeric)
        self.reference = reference
        self.prediction = pd.Series(prediction, index=reference.index)
        self.overall = float(np.mean(prediction))

        self.direction: dict[str, float] = {}
        self.category_effect: dict[str, pd.Series] = {}

        for column in self.features:
            if column in self.numeric:
                values = reference[column]
                if values.nunique() > 1:
                    self.direction[column] = float(
                        pd.Series(values).corr(self.prediction, method="spearman")
                    )
            else:
                self.category_effect[column] = self.prediction.groupby(reference[column]).mean()

    def explain(self, rows: pd.DataFrame, prediction: np.ndarray) -> list[str]:
        texts = []

        for position, (_, row) in enumerate(rows.iterrows()):
            above = prediction[position] >= self.overall
            reasons = []

            for column in self.features:
                reason = self._reason(column, row[column], above)
                if reason:
                    reasons.append(reason)
                if len(reasons) == REASONS_PER_ROW:
                    break

            texts.append("; ".join(reasons))

        return texts

    def _reason(self, column: str, value, above: bool) -> str | None:
        label = humanize(column)

        if column in self.numeric:
            direction = self.direction.get(column)
            if direction is None or pd.isna(value) or abs(direction) < 0.05:
                return None

            # Mid-rank percentile, so a value shared by many rows (e.g. 0)
            # is not reported as an extreme.
            reference = self.reference[column]
            percentile = float((reference < value).mean() + 0.5 * (reference == value).mean())
            high, low = percentile >= HIGH, percentile <= LOW
            pushes_up = (high and direction > 0) or (low and direction < 0)
            pushes_down = (high and direction < 0) or (low and direction > 0)

            if (above and pushes_up) or (not above and pushes_down):
                band = f"top {max(1, round((1 - percentile) * 100))}%" if high else f"bottom {max(1, round(percentile * 100))}%"
                return f"{'high' if high else 'low'} {label} ({format_number(value)}, {band})"
            return None

        effects = self.category_effect.get(column)
        if effects is None or value not in effects.index or not self.overall:
            return None

        ratio = effects[value] / self.overall
        if (above and ratio >= 1.15) or (not above and ratio <= 0.87):
            return f"{label} = {value} ({'higher' if ratio > 1 else 'lower'} than average)"
        return None
