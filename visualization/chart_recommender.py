from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from core.schema_inference import (
    DatasetSchema,
    format_number,
    humanize,
    infer_schema,
    suggest_time_grain,
)

from .chart_schema import ChartRecommendation, ChartSpec, value_label


@dataclass
class _Candidate:
    score: float
    recommendation: ChartRecommendation
    category: str
    columns: tuple[str, ...]
    measure: str | None = None


class ChartRecommender:
    """
    Recommends charts from the dataset schema.

    Responsibilities:
        - Recommend suitable chart types and aggregations
        - Generate a diverse automatic chart set, ranked by how much
          each chart actually reveals in the data
        - Pick a chart for manually selected X/Y columns

    Column semantics come from core.schema_inference; this class never
    re-detects column types. It does NOT execute queries or render charts.
    """

    # Charts per family in the automatic set.
    MAX_PER_CATEGORY = 2

    # Each column may anchor at most this many automatic charts.
    MAX_PER_COLUMN = 2

    # Measures this correlated with an already chosen one are redundant.
    REDUNDANT_CORRELATION = 0.9

    # Sample size used for scoring on large datasets.
    SCORING_SAMPLE = 50_000

    def __init__(self, max_dimension_categories: int = 20):
        self.max_dimension_categories = max_dimension_categories

    # ============================================================
    # PUBLIC API
    # ============================================================

    def recommend(
        self,
        dataframe: pd.DataFrame,
        x_column: Optional[str] = None,
        y_column: Optional[str] = None,
        schema: DatasetSchema | None = None,
    ) -> ChartRecommendation:
        """
        Recommend a single chart for the selected columns.
        Used by the manual chart builder's AUTO mode.
        """

        self._validate_dataframe(dataframe)
        schema = schema or infer_schema(dataframe)

        for column in (x_column, y_column):
            if column is not None and column not in dataframe.columns:
                raise ValueError(
                    f"Column '{column}' does not exist in the dataframe."
                )

        if x_column and y_column:
            if x_column == y_column:
                raise ValueError("X and Y columns must be different.")
            return self._recommend_pair(schema, x_column, y_column)

        column = x_column or y_column

        if column:
            return self._recommend_single(schema, column)

        defaults = self.recommend_default_charts(
            dataframe, max_charts=1, schema=schema,
        )

        if not defaults:
            raise ValueError(
                "Unable to automatically recommend a chart for the dataframe."
            )

        return defaults[0]

    def recommend_default_charts(
        self,
        dataframe: pd.DataFrame,
        max_charts: int = 6,
        schema: DatasetSchema | None = None,
    ) -> list[ChartRecommendation]:
        """
        Automatically generate a diverse set of useful charts.
        """

        self._validate_dataframe(dataframe)

        if max_charts <= 0:
            raise ValueError("max_charts must be greater than 0.")

        schema = schema or infer_schema(dataframe)

        sample = (
            dataframe.sample(self.SCORING_SAMPLE, random_state=0)
            if len(dataframe) > self.SCORING_SAMPLE
            else dataframe
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            correlations = self._measure_correlations(sample, schema)

            candidates: list[_Candidate] = []
            candidates += self._trend_candidates(sample, schema)
            candidates += self._group_candidates(sample, schema)
            candidates += self._relationship_candidates(schema, correlations)
            candidates += self._distribution_candidates(sample, schema)
            candidates += self._composition_candidates(sample, schema)

        candidates.sort(key=lambda item: item.score, reverse=True)

        return self._select_diverse(candidates, correlations, max_charts)

    # ============================================================
    # CANDIDATE GENERATORS
    # ============================================================

    def _trend_candidates(self, dataframe, schema) -> list[_Candidate]:
        time_column = schema.primary_time

        if not time_column:
            return []

        candidates = []

        for measure in schema.aggregatable:
            aggregation = self._default_aggregation(schema, measure)

            monthly = self._trend_strength(
                dataframe, time_column, measure, aggregation,
            )

            if monthly is None:
                continue

            r2, change_pct = monthly

            spec = ChartSpec(
                chart_type="LINE",
                x_column=time_column,
                y_column=measure,
                aggregation=aggregation,
                time_grain="AUTO",
            )

            direction = "up" if change_pct > 0 else "down"

            explanation = (
                f"{value_label(aggregation, measure)} over "
                f"{humanize(time_column)}. "
                + (
                    f"It moves {direction} by about {abs(change_pct):.0f}% "
                    f"across the period (trend fit R² {r2:.2f})."
                    if r2 >= 0.3
                    else "No consistent upward or downward trend."
                )
            )

            candidates.append(_Candidate(
                score=70 + 30 * r2,
                recommendation=ChartRecommendation(
                    chart_spec=spec,
                    confidence=round(0.7 + 0.3 * r2, 2),
                    explanation=explanation,
                    category="trend",
                ),
                category="trend",
                columns=(time_column, measure),
                measure=measure,
            ))

        return candidates

    def _group_candidates(self, dataframe, schema) -> list[_Candidate]:
        candidates = []

        dimensions = [
            name
            for name in schema.groupable
            if 2 <= schema.columns[name].n_unique <= self.max_dimension_categories
        ]

        for dimension in dimensions:
            for measure in schema.aggregatable:

                if measure == dimension:
                    continue

                strength = self._group_effect(dataframe, dimension, measure)

                if strength is None:
                    continue

                eta_squared, top, bottom = strength

                spec = ChartSpec(
                    chart_type="BAR",
                    x_column=dimension,
                    y_column=measure,
                    # Group comparisons use means so group sizes don't
                    # dominate; totals are available in the builder.
                    aggregation="AVERAGE",
                )

                is_rate = measure in schema.numeric_binaries
                fmt = self._format_value

                explanation = (
                    f"{humanize(dimension)} explains {eta_squared:.0%} of "
                    f"the variation in {humanize(measure)}"
                    f"{' rate' if is_rate else ''}. "
                    f"Highest: {top[0]} ({fmt(top[1], is_rate)}), "
                    f"lowest: {bottom[0]} ({fmt(bottom[1], is_rate)})."
                )

                # Rates of a binary outcome are usually the key question.
                bonus = 10 if is_rate else 0

                candidates.append(_Candidate(
                    score=55 + min(45, eta_squared * 300) + bonus,
                    recommendation=ChartRecommendation(
                        chart_spec=spec,
                        confidence=round(min(0.99, 0.6 + eta_squared * 3), 2),
                        explanation=explanation,
                        category="comparison",
                    ),
                    category="comparison",
                    columns=(dimension, measure),
                    measure=measure,
                ))

        return candidates

    def _relationship_candidates(self, schema, correlations) -> list[_Candidate]:
        if correlations is None:
            return []

        candidates = []
        measures = list(correlations.columns)

        for i, x_column in enumerate(measures):
            for y_column in measures[i + 1:]:
                r = correlations.loc[x_column, y_column]

                if pd.isna(r):
                    continue

                strength = abs(float(r))

                # Near-identical columns tell you nothing new.
                if strength < 0.3 or strength >= 0.95:
                    continue

                direction = "positive" if r > 0 else "negative"

                spec = ChartSpec(
                    chart_type="SCATTER",
                    x_column=x_column,
                    y_column=y_column,
                    aggregation="NONE",
                )

                candidates.append(_Candidate(
                    score=50 + strength * 45,
                    recommendation=ChartRecommendation(
                        chart_spec=spec,
                        confidence=round(strength, 2),
                        explanation=(
                            f"{humanize(x_column)} and {humanize(y_column)} "
                            f"have a {direction} correlation of {r:.2f}."
                        ),
                        category="relationship",
                    ),
                    category="relationship",
                    columns=(x_column, y_column),
                ))

        return candidates

    def _distribution_candidates(self, dataframe, schema) -> list[_Candidate]:
        candidates = []

        for measure in schema.measures:
            values = dataframe[measure].dropna()

            if values.nunique() < 5:
                continue

            skew = float(values.skew())

            if not np.isfinite(skew):
                skew = 0.0

            shape = (
                "roughly symmetric"
                if abs(skew) < 0.5
                else f"{'right' if skew > 0 else 'left'}-skewed"
            )

            spec = ChartSpec(
                chart_type="HISTOGRAM",
                x_column=measure,
                aggregation="NONE",
            )

            candidates.append(_Candidate(
                score=40 + min(25, abs(skew) * 8),
                recommendation=ChartRecommendation(
                    chart_spec=spec,
                    confidence=0.8,
                    explanation=(
                        f"{humanize(measure)} is {shape}: median "
                        f"{format_number(values.median())}, middle 50% "
                        f"between {format_number(values.quantile(0.25))} "
                        f"and {format_number(values.quantile(0.75))}."
                    ),
                    category="distribution",
                ),
                category="distribution",
                columns=(measure,),
                measure=measure,
            ))

        return candidates

    def _composition_candidates(self, dataframe, schema) -> list[_Candidate]:
        candidates = []

        for dimension in schema.groupable:
            counts = dataframe[dimension].value_counts(dropna=True)

            if not 2 <= len(counts) <= self.max_dimension_categories:
                continue

            share = counts.iloc[0] / counts.sum()
            chart_type = "PIE" if len(counts) <= 5 else "BAR"

            spec = ChartSpec(
                chart_type=chart_type,
                x_column=dimension,
                aggregation="COUNT",
                title=f"Rows by {humanize(dimension)}",
            )

            # Uneven splits are more informative than uniform ones.
            evenness = 1 - abs(share - 1 / len(counts))

            candidates.append(_Candidate(
                score=35 + (1 - evenness) * 30,
                recommendation=ChartRecommendation(
                    chart_spec=spec,
                    confidence=0.75,
                    explanation=(
                        f"{len(counts)} {humanize(dimension)} values; the "
                        f"largest, {counts.index[0]}, holds {share:.0%} of rows."
                    ),
                    category="composition",
                ),
                category="composition",
                columns=(dimension,),
            ))

        return candidates

    # ============================================================
    # SELECTION
    # ============================================================

    def _select_diverse(
        self,
        candidates: list[_Candidate],
        correlations: pd.DataFrame | None,
        max_charts: int,
    ) -> list[ChartRecommendation]:

        selected: list[_Candidate] = []
        category_counts: dict[str, int] = {}
        column_counts: dict[str, int] = {}

        def correlated(a: str, b: str) -> bool:
            if a == b:
                return True
            r = self._lookup_correlation(correlations, a, b)
            return r is not None and abs(r) >= self.REDUNDANT_CORRELATION

        def is_redundant(candidate: _Candidate) -> bool:
            if correlations is None:
                return False

            # Two scatter plots whose axes are (near-)duplicates of each
            # other show the same picture.
            if candidate.category == "relationship":
                x_new, y_new = candidate.columns
                for chosen in selected:
                    if chosen.category != "relationship":
                        continue
                    x_old, y_old = chosen.columns
                    if (
                        (correlated(x_new, x_old) and correlated(y_new, y_old))
                        or (correlated(x_new, y_old) and correlated(y_new, x_old))
                    ):
                        return True
                return False

            if candidate.measure is None:
                return False

            for chosen in selected:
                if chosen.category != candidate.category:
                    continue
                if chosen.measure is None:
                    continue
                if chosen.measure == candidate.measure:
                    continue

                r = self._lookup_correlation(
                    correlations, chosen.measure, candidate.measure,
                )

                if r is not None and abs(r) >= self.REDUNDANT_CORRELATION:
                    return True

            return False

        def fits(candidate: _Candidate, strict: bool) -> bool:
            if any(c is candidate for c in selected):
                return False

            if category_counts.get(candidate.category, 0) >= (
                self.MAX_PER_CATEGORY if strict else max_charts
            ):
                return False

            if strict and any(
                column_counts.get(column, 0) >= self.MAX_PER_COLUMN
                for column in candidate.columns
            ):
                return False

            return not is_redundant(candidate)

        for strict in (True, False):
            for candidate in candidates:
                if len(selected) >= max_charts:
                    break

                if not fits(candidate, strict):
                    continue

                selected.append(candidate)
                category_counts[candidate.category] = (
                    category_counts.get(candidate.category, 0) + 1
                )
                for column in candidate.columns:
                    column_counts[column] = column_counts.get(column, 0) + 1

        return [candidate.recommendation for candidate in selected]

    # ============================================================
    # SINGLE / PAIR RECOMMENDATION (manual AUTO mode)
    # ============================================================

    def _recommend_pair(
        self,
        schema: DatasetSchema,
        x_column: str,
        y_column: str,
    ) -> ChartRecommendation:

        x_role = schema.role_of(x_column)
        y_numeric = y_column in schema.aggregatable
        x_numeric = x_column in schema.aggregatable

        if x_role == "time" and y_numeric:
            spec = ChartSpec(
                chart_type="LINE",
                x_column=x_column,
                y_column=y_column,
                aggregation=self._default_aggregation(schema, y_column),
                time_grain="AUTO",
            )
            return self._wrap(spec, "Time on X with a numeric measure: line chart.")

        if x_role == "time":
            spec = ChartSpec(
                chart_type="LINE",
                x_column=x_column,
                y_column=y_column,
                aggregation="COUNT_DISTINCT",
                time_grain="AUTO",
            )
            return self._wrap(spec, "Distinct values of the second column over time.")

        if x_role == "measure" and y_numeric:
            spec = ChartSpec(
                chart_type="SCATTER",
                x_column=x_column,
                y_column=y_column,
                aggregation="NONE",
            )
            return self._wrap(spec, "Two numeric columns: scatter plot.")

        if y_numeric:
            spec = ChartSpec(
                chart_type="BAR",
                x_column=x_column,
                y_column=y_column,
                aggregation=self._default_aggregation(schema, y_column),
            )
            return self._wrap(spec, "Categories on X with a numeric measure: bar chart.")

        if x_numeric and schema.role_of(y_column) in {"dimension", "binary"}:
            spec = ChartSpec(
                chart_type="BOX",
                x_column=y_column,
                y_column=x_column,
            )
            return self._wrap(spec, "Numeric values split by category: box plot.")

        spec = ChartSpec(
            chart_type="BAR",
            x_column=x_column,
            y_column=y_column,
            aggregation="COUNT_DISTINCT",
        )
        return self._wrap(spec, "Distinct values of Y per category of X.")

    def _recommend_single(
        self,
        schema: DatasetSchema,
        column: str,
    ) -> ChartRecommendation:

        role = schema.role_of(column)

        if role == "measure":
            spec = ChartSpec(chart_type="HISTOGRAM", x_column=column)
            return self._wrap(spec, "Numeric column: histogram of its distribution.")

        if role == "time":
            spec = ChartSpec(
                chart_type="LINE",
                x_column=column,
                aggregation="COUNT",
                time_grain="AUTO",
            )
            return self._wrap(spec, "Number of rows over time.")

        spec = ChartSpec(chart_type="BAR", x_column=column, aggregation="COUNT")
        return self._wrap(spec, "Number of rows per category.")

    @staticmethod
    def _wrap(spec: ChartSpec, explanation: str) -> ChartRecommendation:
        return ChartRecommendation(
            chart_spec=spec,
            confidence=0.8,
            explanation=explanation,
            category="manual",
        )

    # ============================================================
    # STATISTICS
    # ============================================================

    @staticmethod
    def default_aggregation(schema: DatasetSchema, measure: str | None) -> str:
        """
        Sensible default aggregation for a column:
            binary 0/1  -> AVERAGE (a rate)
            count-like  -> SUM
            identifier / text -> COUNT_DISTINCT
            otherwise   -> AVERAGE
        """

        if measure is None:
            return "COUNT"

        if measure in schema.numeric_binaries:
            return "AVERAGE"

        column = schema.columns.get(measure)

        if column is None or not column.is_numeric:
            return "COUNT_DISTINCT"

        if column.role == "identifier":
            return "COUNT_DISTINCT"

        if column.kind == "count":
            return "SUM"

        return "AVERAGE"

    _default_aggregation = default_aggregation

    def _measure_correlations(self, dataframe, schema) -> pd.DataFrame | None:
        measures = [
            name
            for name in schema.measures
            if dataframe[name].nunique() > 2
        ]

        if len(measures) < 2:
            return None

        return dataframe[measures].corr(method="pearson")

    @staticmethod
    def _lookup_correlation(correlations, a, b) -> float | None:
        if a in correlations.index and b in correlations.columns:
            value = correlations.loc[a, b]
            return None if pd.isna(value) else float(value)
        return None

    @staticmethod
    def _group_effect(dataframe, dimension, measure):
        """
        Eta-squared: share of the measure's variance explained by the
        grouping. 0 = groups look identical, 1 = groups fully separate.
        """

        data = dataframe[[dimension, measure]].dropna()

        if len(data) < 10:
            return None

        total_variance = data[measure].var(ddof=0)

        if not total_variance or not np.isfinite(total_variance):
            return None

        groups = data.groupby(dimension, observed=True)[measure]
        means = groups.mean()
        sizes = groups.size()

        if len(means) < 2:
            return None

        grand_mean = data[measure].mean()

        between = float((sizes * (means - grand_mean) ** 2).sum() / len(data))

        eta_squared = between / total_variance

        # Ignore groups too small to trust when naming extremes.
        reliable = means[sizes >= max(3, len(data) * 0.01)]

        if reliable.empty:
            reliable = means

        top = (reliable.idxmax(), float(reliable.max()))
        bottom = (reliable.idxmin(), float(reliable.min()))

        return eta_squared, top, bottom

    @staticmethod
    def _trend_strength(dataframe, time_column, measure, aggregation):
        data = dataframe[[time_column, measure]].dropna()

        if data.empty:
            return None

        grain = suggest_time_grain(data[time_column])
        periods = (
            data[time_column].dt.to_period(grain)
            if grain
            else data[time_column]
        )

        series = (
            data.groupby(periods)[measure]
            .agg("sum" if aggregation == "SUM" else "mean")
        )

        if len(series) < 3:
            return None

        y = series.to_numpy(dtype=float)
        x = np.arange(len(y))

        if np.std(y) == 0:
            return 0.0, 0.0

        slope, _ = np.polyfit(x, y, 1)
        r = np.corrcoef(x, y)[0, 1]

        mean = np.mean(np.abs(y))
        change_pct = (slope * (len(y) - 1) / mean * 100) if mean else 0.0

        return float(r ** 2), float(change_pct)

    @staticmethod
    def _format_value(value: float, is_rate: bool) -> str:
        if is_rate:
            return f"{value:.1%}"
        return format_number(value)

    # ============================================================
    # VALIDATION
    # ============================================================

    def _validate_dataframe(
        self,
        dataframe: pd.DataFrame,
    ) -> None:

        if dataframe is None:
            raise ValueError("DataFrame cannot be None.")

        if dataframe.empty:
            raise ValueError("DataFrame cannot be empty.")
