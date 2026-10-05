"""
Target-aware EDA for a training table.

    DataFrame + schema + target column
        -> target summary      (positive rate, unlabelled rows, imbalance)
        -> segments            (target rate per category, chi-square, Cramér's V)
        -> feature signal      (univariate AUC / correlation per numeric column)
        -> leakage checks      (near-perfect predictors, target copies,
                                future-sounding names, dates past the snapshot)
        -> panel / time split  (entity x period structure, split advice, drift)

Nothing here knows a dataset: the target is proposed from column roles,
values and naming conventions, and every check runs on whatever columns
the table has.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from core.schema_inference import (
    DatasetSchema,
    format_number,
    humanize,
    name_tokens,
    suggest_time_grain,
)


# Name words that suggest a column is the outcome to predict.
TARGET_NAME_TOKENS = {
    "target", "label", "outcome", "next", "future", "churn", "churned",
    "converted", "conversion", "will", "class", "response", "default",
    "fraud", "flag", "result", "y",
}

# Name words that suggest a feature holds information from after the
# moment of prediction.
FUTURE_NAME_TOKENS = {
    "next", "future", "following", "after", "later", "outcome", "result",
    "target", "label",
}

# Imbalance thresholds on the minority class share.
CRITICAL_MINORITY = 0.02
WARNING_MINORITY = 0.10

# A single feature this predictive is almost certainly leaking.
LEAK_AUC = 0.98
LEAK_CRAMERS_V = 0.9

# Groups smaller than this are folded into "Other" for segment stats.
MIN_GROUP_ROWS = 20

# Share of periods used for training in the suggested time split.
TRAIN_SHARE = 0.8


# ----------------------------------------------------------------------
# Results
# ----------------------------------------------------------------------

@dataclass
class TargetCandidate:
    column: str
    kind: str          # binary | numeric
    score: float
    reason: str


@dataclass
class Finding:
    severity: str      # critical | warning | info
    title: str
    detail: str
    column: str | None = None


@dataclass
class SegmentResult:
    column: str
    table: pd.DataFrame          # group, rows, positives / mean, rate, lift
    cramers_v: float | None
    p_value: float | None
    highest: tuple[Any, float]
    lowest: tuple[Any, float]


@dataclass
class FeatureResult:
    column: str
    auc: float | None            # binary targets
    correlation: float | None    # numeric targets (Spearman)
    strength: float              # 0..1, comparable across features
    direction: str               # "higher -> more", "higher -> less"
    mean_positive: float | None
    mean_negative: float | None
    bins: pd.DataFrame | None    # target rate by quartile


@dataclass
class PanelInfo:
    entity: str
    time: str
    n_entities: int
    n_periods: int
    rows_per_entity: float


@dataclass
class SplitAdvice:
    method: str                  # time | stratified | random
    summary: str
    table: pd.DataFrame | None = None


@dataclass
class TargetReport:
    target: str
    kind: str                    # binary | numeric
    positive_label: Any
    n_rows: int
    n_labelled: int
    positive_rate: float | None  # binary
    mean: float | None           # numeric
    class_counts: pd.DataFrame | None
    imbalance: Finding | None
    segments: list[SegmentResult] = field(default_factory=list)
    features: list[FeatureResult] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    excluded: list[tuple[str, str]] = field(default_factory=list)
    panel: PanelInfo | None = None
    time_column: str | None = None
    time_grain: str | None = None
    over_time: pd.DataFrame | None = None
    split: SplitAdvice | None = None


# ----------------------------------------------------------------------
# Target detection
# ----------------------------------------------------------------------

def detect_targets(
    dataframe: pd.DataFrame,
    schema: DatasetSchema,
) -> list[TargetCandidate]:
    """
    Columns that could be the outcome to predict, most likely first.
    Two-valued columns are preferred; names such as "NextMonthOrder",
    "churned" or "label" and a position near the end add weight.
    """

    columns = list(dataframe.columns)
    candidates = []

    for position, column in enumerate(columns):
        role = schema.role_of(column)
        tokens = set(name_tokens(column))
        named = bool(tokens & TARGET_NAME_TOKENS)

        if role == "binary":
            kind = "binary"
            score = 2.0
            reasons = ["two distinct values"]
        elif role == "measure" and named:
            kind = "numeric"
            score = 1.0
            reasons = ["numeric"]
        else:
            continue

        if named:
            score += 3.0
            reasons.append("name suggests an outcome")

        if kind == "binary":
            minority = _minority_share(dataframe[column])
            if minority is not None and 0.01 <= minority <= 0.5:
                score += 0.5
            if schema.columns[column].is_numeric:
                score += 0.5
                reasons.append("coded 0/1")

        if position >= len(columns) * 2 / 3:
            score += 0.5

        candidates.append(
            TargetCandidate(column, kind, score, ", ".join(reasons))
        )

    return sorted(candidates, key=lambda candidate: -candidate.score)


def eligible_targets(dataframe: pd.DataFrame, schema: DatasetSchema) -> list[str]:
    """Every column that can be analysed as a target."""

    return [
        column
        for column in dataframe.columns
        if schema.role_of(column) in {"binary", "measure"}
    ]


# ----------------------------------------------------------------------
# Analysis
# ----------------------------------------------------------------------

def analyze_target(
    dataframe: pd.DataFrame,
    schema: DatasetSchema,
    target: str,
) -> TargetReport:

    if target not in dataframe.columns:
        raise ValueError(f"Column '{target}' does not exist in the dataset.")

    role = schema.role_of(target)

    if role == "binary":
        kind = "binary"
        y, positive = encode_binary(dataframe[target])
    elif role == "measure":
        kind = "numeric"
        y, positive = pd.to_numeric(dataframe[target], errors="coerce"), None
    else:
        raise ValueError(
            f"'{target}' can't be used as a target: choose a two-valued "
            f"or numeric column."
        )

    labelled = y.notna()

    report = TargetReport(
        target=target,
        kind=kind,
        positive_label=positive,
        n_rows=len(dataframe),
        n_labelled=int(labelled.sum()),
        positive_rate=float(y[labelled].mean()) if kind == "binary" else None,
        mean=float(y[labelled].mean()) if kind == "numeric" else None,
        class_counts=_class_counts(dataframe[target]) if kind == "binary" else None,
        imbalance=None,
    )

    if report.n_labelled == 0:
        raise ValueError(f"'{target}' has no values to analyse.")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        if kind == "binary":
            report.imbalance = _imbalance(report, positive)

        report.excluded = _excluded_columns(dataframe, schema, target)
        excluded = {column for column, _ in report.excluded}

        report.segments = _segments(dataframe, schema, target, y, kind, excluded)
        report.features = _features(dataframe, schema, target, y, kind, excluded)

        report.time_column = snapshot_time(dataframe, schema)
        report.panel = _panel(dataframe, schema, report.time_column)

        if report.time_column:
            report.time_grain = suggest_time_grain(dataframe[report.time_column])
            report.over_time = _over_time(dataframe, report.time_column, report.time_grain, y)

        report.findings = _findings(dataframe, schema, report, y)
        report.split = _split_advice(dataframe, report, y)

    return report


# ----------------------------------------------------------------------
# Target helpers
# ----------------------------------------------------------------------

_POSITIVE_WORDS = {"1", "true", "yes", "y", "t", "positive", "pos"}


def encode_binary(series: pd.Series) -> tuple[pd.Series, Any]:
    """1.0 for the positive class, 0.0 otherwise, NaN where missing."""

    values = list(pd.Series(series.dropna().unique()))

    if len(values) != 2:
        raise ValueError("A binary target needs exactly two distinct values.")

    positive = next(
        (value for value in values if str(value).strip().lower() in _POSITIVE_WORDS),
        None,
    )

    if positive is None:
        # Otherwise the rarer value is the event of interest.
        counts = series.value_counts()
        positive = counts.index[-1]

    encoded = (series == positive).astype(float)
    encoded[series.isna()] = np.nan

    return encoded, positive


def _minority_share(series: pd.Series) -> float | None:
    counts = series.value_counts(normalize=True)
    if len(counts) != 2:
        return None
    return float(counts.min())


def _class_counts(series: pd.Series) -> pd.DataFrame:
    counts = series.value_counts(dropna=False)
    return pd.DataFrame({
        "Value": counts.index.astype(str),
        "Rows": counts.values,
        "Share": (counts.values / counts.sum()).round(4),
    })


def _imbalance(report: TargetReport, positive: Any) -> Finding:
    rate = report.positive_rate
    minority = min(rate, 1 - rate)
    share = f"{minority:.1%}"

    if minority < CRITICAL_MINORITY:
        severity, title = "critical", "Severe class imbalance"
    elif minority < WARNING_MINORITY:
        severity, title = "warning", "Class imbalance"
    else:
        return Finding(
            "info",
            "Classes are reasonably balanced",
            f"The minority class is {share} of labelled rows.",
            report.target,
        )

    return Finding(
        severity,
        title,
        f"Only {share} of labelled rows are in the minority class. Use a "
        f"stratified split, class weights or resampling, and judge models "
        f"by recall, precision and PR-AUC rather than accuracy (always "
        f"predicting the majority class would already score "
        f"{1 - minority:.1%}).",
        report.target,
    )


def _excluded_columns(
    dataframe: pd.DataFrame,
    schema: DatasetSchema,
    target: str,
) -> list[tuple[str, str]]:
    """Columns that should not be fed to a model as they are."""

    excluded = []

    for column in dataframe.columns:
        if column == target:
            continue

        role = schema.role_of(column)

        if role == "identifier":
            excluded.append((column, "ID / key: identifies rows, carries no general signal"))
        elif role in {"constant", "empty"}:
            excluded.append((column, "has a single value or none"))
        elif role == "text":
            excluded.append((column, "free text: needs separate text features"))
        elif role == "time":
            excluded.append((column, "date: use for the split, or derive features such as days since"))

    return excluded


# ----------------------------------------------------------------------
# Segments
# ----------------------------------------------------------------------

def _segments(
    dataframe: pd.DataFrame,
    schema: DatasetSchema,
    target: str,
    y: pd.Series,
    kind: str,
    excluded: set[str],
) -> list[SegmentResult]:

    results = []
    labelled = y.notna()
    overall = float(y[labelled].mean())

    for column in schema.groupable:
        if column == target or column in excluded:
            continue

        groups = _fold_small_groups(dataframe.loc[labelled, column])
        frame = pd.DataFrame({"group": groups, "y": y[labelled]})

        summary = frame.groupby("group", dropna=False)["y"].agg(["count", "sum", "mean"])

        if len(summary) < 2:
            continue

        table = pd.DataFrame({
            column: summary.index.astype(str),
            "Rows": summary["count"].astype(int).values,
        })

        if kind == "binary":
            table["Positives"] = summary["sum"].astype(int).values
            table["Rate"] = summary["mean"].round(4).values
            cramers_v, p_value = _chi_square(summary)
        else:
            table["Mean"] = summary["mean"].round(4).values
            cramers_v, p_value = _anova(frame)

        value_column = "Rate" if kind == "binary" else "Mean"
        table["Lift"] = (table[value_column] / overall).round(2) if overall else np.nan
        table = table.sort_values(value_column, ascending=False).reset_index(drop=True)

        results.append(
            SegmentResult(
                column=column,
                table=table,
                cramers_v=cramers_v,
                p_value=p_value,
                highest=(table.iloc[0][column], float(table.iloc[0][value_column])),
                lowest=(table.iloc[-1][column], float(table.iloc[-1][value_column])),
            )
        )

    return sorted(results, key=lambda result: -(result.cramers_v or 0))


def _fold_small_groups(series: pd.Series) -> pd.Series:
    counts = series.value_counts()
    small = counts[counts < MIN_GROUP_ROWS].index

    if len(small) == 0:
        return series

    return series.astype(object).where(~series.isin(small), "Other")


def _chi_square(summary: pd.DataFrame) -> tuple[float | None, float | None]:
    positives = summary["sum"].to_numpy()
    negatives = summary["count"].to_numpy() - positives
    table = np.vstack([positives, negatives])

    if (table.sum(axis=1) == 0).any():
        return None, None

    chi2, p_value, _, _ = stats.chi2_contingency(table)
    n = table.sum()
    k = min(table.shape) - 1
    cramers_v = math.sqrt(chi2 / (n * k)) if n and k else None

    return cramers_v, float(p_value)


def _anova(frame: pd.DataFrame) -> tuple[float | None, float | None]:
    samples = [group["y"].to_numpy() for _, group in frame.groupby("group") if len(group) > 1]

    if len(samples) < 2:
        return None, None

    _, p_value = stats.f_oneway(*samples)

    # Effect size: correlation ratio (eta), comparable to Cramér's V.
    grand = frame["y"].mean()
    between = sum(len(sample) * (sample.mean() - grand) ** 2 for sample in samples)
    total = ((frame["y"] - grand) ** 2).sum()
    eta = math.sqrt(between / total) if total else None

    return eta, float(p_value)


# ----------------------------------------------------------------------
# Numeric features
# ----------------------------------------------------------------------

def _features(
    dataframe: pd.DataFrame,
    schema: DatasetSchema,
    target: str,
    y: pd.Series,
    kind: str,
    excluded: set[str],
) -> list[FeatureResult]:

    results = []

    for column in schema.aggregatable:
        if column == target or column in excluded:
            continue

        x = pd.to_numeric(dataframe[column], errors="coerce")
        mask = x.notna() & y.notna()

        if mask.sum() < 10 or x[mask].nunique() < 2:
            continue

        xs, ys = x[mask], y[mask]

        if kind == "binary":
            positives, negatives = xs[ys == 1], xs[ys == 0]

            if positives.empty or negatives.empty:
                continue

            u, _ = stats.mannwhitneyu(positives, negatives, alternative="two-sided")
            auc = float(u / (len(positives) * len(negatives)))

            results.append(
                FeatureResult(
                    column=column,
                    auc=auc,
                    correlation=None,
                    strength=abs(auc - 0.5) * 2,
                    direction="higher → more" if auc >= 0.5 else "higher → less",
                    mean_positive=float(positives.mean()),
                    mean_negative=float(negatives.mean()),
                    bins=_quartile_rates(xs, ys),
                )
            )
        else:
            correlation = float(stats.spearmanr(xs, ys).statistic)

            if math.isnan(correlation):
                continue

            results.append(
                FeatureResult(
                    column=column,
                    auc=None,
                    correlation=correlation,
                    strength=abs(correlation),
                    direction="higher → more" if correlation >= 0 else "higher → less",
                    mean_positive=None,
                    mean_negative=None,
                    bins=_quartile_rates(xs, ys),
                )
            )

    return sorted(results, key=lambda result: -result.strength)


def _quartile_rates(x: pd.Series, y: pd.Series) -> pd.DataFrame | None:
    try:
        bins = pd.qcut(x, 4, duplicates="drop")
    except ValueError:
        return None

    if bins.nunique() < 2:
        return None

    summary = y.groupby(bins, observed=True).agg(["count", "mean"])

    return pd.DataFrame({
        "Range": [str(interval) for interval in summary.index],
        "Rows": summary["count"].astype(int).values,
        "Target": summary["mean"].round(4).values,
    })


# ----------------------------------------------------------------------
# Panel structure and time
# ----------------------------------------------------------------------

def snapshot_time(dataframe: pd.DataFrame, schema: DatasetSchema) -> str | None:
    """
    The date each row describes. A column recorded at a regular period
    (one value per month, week, ...) is preferred over event dates such
    as "closed on", which vary row by row.
    """

    regular = [column for column in schema.times if native_grain(dataframe[column])]

    if not regular:
        return schema.primary_time

    # Snapshots sit on period boundaries (the 1st of each month), while
    # event dates that happen to repeat monthly do not.
    def rank(column: str) -> tuple[bool, bool]:
        values = dataframe[column].dropna()
        start = values.dt.to_period(native_grain(values)).dt.start_time
        aligned = (values == start).mean() >= 0.95
        return aligned, column == schema.primary_time

    return max(regular, key=rank)


def _panel(
    dataframe: pd.DataFrame,
    schema: DatasetSchema,
    time: str | None,
) -> PanelInfo | None:
    """
    An entity observed once per period (doctor x month): a key column
    whose values repeat, and whose (key, time) pairs are unique.
    """

    if time is None:
        return None

    n_rows = len(dataframe)

    for column in schema.identifiers:
        n_entities = dataframe[column].nunique()

        if n_entities < 2 or n_entities > n_rows * 0.9:
            continue

        pairs = dataframe[[column, time]].dropna().drop_duplicates()

        if len(pairs) < n_rows * 0.95:
            continue

        return PanelInfo(
            entity=column,
            time=time,
            n_entities=int(n_entities),
            n_periods=int(dataframe[time].nunique()),
            rows_per_entity=round(n_rows / n_entities, 1),
        )

    return None


def _periods(series: pd.Series, grain: str | None) -> pd.Series:
    if grain:
        return series.dt.to_period(grain).dt.start_time
    return series


def _over_time(
    dataframe: pd.DataFrame,
    time: str,
    grain: str | None,
    y: pd.Series,
) -> pd.DataFrame | None:

    periods = _periods(dataframe[time], grain)
    frame = pd.DataFrame({"Period": periods, "y": y}).dropna(subset=["Period"])

    if frame["Period"].nunique() < 2:
        return None

    summary = frame.groupby("Period").agg(
        Rows=("y", "size"),
        Labelled=("y", "count"),
        Target=("y", "mean"),
    )
    summary["Target"] = summary["Target"].round(4)

    return summary.reset_index()


# ----------------------------------------------------------------------
# Findings (leakage, labels, drift)
# ----------------------------------------------------------------------

def _findings(
    dataframe: pd.DataFrame,
    schema: DatasetSchema,
    report: TargetReport,
    y: pd.Series,
) -> list[Finding]:

    findings = []
    target = report.target

    for feature in report.features:
        if feature.auc is not None and (feature.auc >= LEAK_AUC or feature.auc <= 1 - LEAK_AUC):
            findings.append(Finding(
                "critical",
                f"Possible leakage: {humanize(feature.column)}",
                f"On its own it separates the classes almost perfectly "
                f"(AUC {feature.auc:.3f}). Check it is known before the "
                f"outcome; if not, drop it.",
                feature.column,
            ))
        elif feature.correlation is not None and abs(feature.correlation) >= LEAK_AUC:
            findings.append(Finding(
                "critical",
                f"Possible leakage: {humanize(feature.column)}",
                f"Correlation with the target is {feature.correlation:.3f}.",
                feature.column,
            ))

    for segment in report.segments:
        if segment.cramers_v is not None and segment.cramers_v >= LEAK_CRAMERS_V:
            findings.append(Finding(
                "critical",
                f"Possible leakage: {humanize(segment.column)}",
                f"Its categories almost determine the target "
                f"(Cramér's V {segment.cramers_v:.2f}).",
                segment.column,
            ))

    labelled = y.notna()

    for column in dataframe.columns:
        if column == target:
            continue

        if schema.columns[column].is_numeric and report.kind == "binary":
            x = pd.to_numeric(dataframe[column], errors="coerce")
            both = labelled & x.notna()
            if both.sum() and x[both].isin([0, 1]).all():
                same = (x[both] == y[both]).mean()
                if same >= 0.99 or same <= 0.01:
                    findings.append(Finding(
                        "critical",
                        f"Copy of the target: {humanize(column)}",
                        "It matches the target (or its opposite) on "
                        "almost every row. Drop it.",
                        column,
                    ))
                    continue

        tokens = set(name_tokens(column))
        if tokens & FUTURE_NAME_TOKENS:
            findings.append(Finding(
                "warning",
                f"Name suggests future information: {humanize(column)}",
                "Make sure its value is known at prediction time.",
                column,
            ))

    findings.extend(_dates_after_snapshot(dataframe, schema, report))

    missing = report.n_rows - report.n_labelled
    if missing:
        findings.append(Finding(
            "warning",
            "Rows without a target value",
            f"{format_number(missing)} rows have no target. These are "
            f"often the latest period whose outcome is not known yet; "
            f"keep them out of training and use them for scoring.",
            target,
        ))

    if report.over_time is not None and len(report.over_time) >= 3:
        rates = report.over_time["Target"].dropna()
        last = report.over_time.iloc[-1]
        if report.kind == "binary" and last["Labelled"] and last["Target"] == 0 and rates.iloc[:-1].mean() > 0:
            findings.append(Finding(
                "warning",
                "Latest period has no positives",
                "The outcome of the last period may not be known yet. "
                "Check before training on it.",
                target,
            ))

        spread = rates.max() - rates.min()
        if report.kind == "binary" and spread >= 0.05:
            findings.append(Finding(
                "info",
                "Target rate changes over time",
                f"It ranges from {rates.min():.1%} to {rates.max():.1%} "
                f"across periods. A time-based split shows how a model "
                f"copes with that drift.",
                target,
            ))

    if report.imbalance is not None and report.imbalance.severity != "info":
        findings.append(report.imbalance)

    order = {"critical": 0, "warning": 1, "info": 2}
    return sorted(findings, key=lambda finding: order[finding.severity])


def _dates_after_snapshot(
    dataframe: pd.DataFrame,
    schema: DatasetSchema,
    report: TargetReport,
) -> list[Finding]:
    """
    In a periodic table (one row per period), other date columns
    should not fall after the end of the row's period: such values were
    recorded later than the moment the row describes.
    """

    time = report.time_column
    grain = native_grain(dataframe[time]) if time else None

    if grain is None:
        return []

    period_end = dataframe[time].dt.to_period(grain).dt.end_time
    findings = []

    for column in schema.times:
        if column == time:
            continue

        values = dataframe[column]
        both = values.notna() & period_end.notna()

        if not both.any():
            continue

        later = (values[both] > period_end[both]).mean()

        if later >= 0.01:
            findings.append(Finding(
                "warning",
                f"Dates after the snapshot period: {humanize(column)}",
                f"{later:.1%} of rows have {humanize(column)} after the end "
                f"of their {humanize(time)} period, so it may hold "
                f"information from the future.",
                column,
            ))

    return findings


def native_grain(series: pd.Series) -> str | None:
    """
    The period a date column is recorded at, from the typical gap
    between its distinct values: one row per day, week, month, quarter
    or year. None for irregular timestamps.
    """

    values = pd.Series(pd.to_datetime(series, errors="coerce").dropna().unique()).sort_values()

    if len(values) < 3:
        return None

    gap = values.diff().dt.days.median()

    for grain, low, high in (
        ("D", 1, 1),
        ("W", 7, 7),
        ("M", 28, 31),
        ("Q", 89, 92),
        ("Y", 365, 366),
    ):
        if low <= gap <= high:
            return grain

    return None


# ----------------------------------------------------------------------
# Train / test split advice
# ----------------------------------------------------------------------

def _split_advice(
    dataframe: pd.DataFrame,
    report: TargetReport,
    y: pd.Series,
) -> SplitAdvice:

    labelled = y.notna()
    time = report.time_column

    if time and dataframe[time].nunique() >= 3:
        periods = _periods(dataframe[time], report.time_grain)
        ordered = sorted(periods[labelled].dropna().unique())
        cutoff = ordered[max(0, math.ceil(len(ordered) * TRAIN_SHARE) - 1)]

        train = labelled & (periods <= cutoff)
        test = labelled & (periods > cutoff)

        table = pd.DataFrame([
            _split_row("Train", periods, train, y),
            _split_row("Test", periods, test, y),
        ])

        summary = (
            f"Split by time: train on {humanize(time)} up to "
            f"{_format_period(cutoff)}, test on the later periods. "
        )

        if report.panel is not None:
            panel = report.panel
            overlap = _random_split_overlap(dataframe[panel.entity][labelled])
            summary += (
                f"Each {humanize(panel.entity)} appears in "
                f"{format_number(panel.n_periods)} periods, so a random "
                f"split would put about {overlap:.0%} of them in both "
                f"train and test and overstate accuracy. "
            )

        if report.kind == "binary":
            summary += "Within the training periods, keep the class mix with a stratified validation split."

        return SplitAdvice("time", summary.strip(), table)

    if report.kind == "binary":
        return SplitAdvice(
            "stratified",
            f"No date column: use a stratified random split "
            f"({TRAIN_SHARE:.0%} / {1 - TRAIN_SHARE:.0%}) so both parts keep the "
            f"{report.positive_rate:.1%} positive rate.",
        )

    return SplitAdvice(
        "random",
        f"No date column: use a random {TRAIN_SHARE:.0%} / {1 - TRAIN_SHARE:.0%} split.",
    )


def _split_row(name: str, periods: pd.Series, mask: pd.Series, y: pd.Series) -> dict[str, Any]:
    chosen = periods[mask]
    return {
        "Part": name,
        "From": _format_period(chosen.min()) if len(chosen) else "",
        "To": _format_period(chosen.max()) if len(chosen) else "",
        "Rows": int(mask.sum()),
        "Target": round(float(y[mask].mean()), 4) if mask.any() else None,
    }


def _random_split_overlap(entities: pd.Series) -> float:
    """Share of entities that a random 80/20 row split puts on both sides."""

    counts = entities.value_counts()
    # An entity with n rows lands only in train with p = share^n.
    in_both = 1 - TRAIN_SHARE ** counts - (1 - TRAIN_SHARE) ** counts
    return float(in_both.mean())


def _format_period(value: Any) -> str:
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).strftime("%Y-%m-%d")
    return str(value)


# ----------------------------------------------------------------------
# Markdown report
# ----------------------------------------------------------------------

def report_markdown(report: TargetReport, dataset_name: str = "") -> str:
    lines = [f"# Target EDA report: {report.target}", ""]

    if dataset_name:
        lines += [f"Dataset: {dataset_name}", ""]

    lines += ["## Summary", ""]
    lines.append(f"- Rows: {format_number(report.n_rows)} ({format_number(report.n_labelled)} with a target)")

    if report.kind == "binary":
        lines.append(f"- Positive class: {report.positive_label}")
        lines.append(f"- Positive rate: {report.positive_rate:.2%}")
    else:
        lines.append(f"- Mean: {format_number(report.mean)}")

    if report.panel:
        panel = report.panel
        lines.append(
            f"- Panel data: {format_number(panel.n_entities)} × "
            f"{humanize(panel.entity)} over {format_number(panel.n_periods)} "
            f"periods of {humanize(panel.time)}"
        )

    lines += ["", "## Warnings and checks", ""]
    for finding in report.findings or []:
        lines.append(f"- **{finding.severity.upper()}: {finding.title}.** {finding.detail}")
    if not report.findings:
        lines.append("- None.")

    if report.split:
        lines += ["", "## Train / test split", "", report.split.summary, ""]
        if report.split.table is not None:
            lines += [_markdown_table(report.split.table), ""]

    lines += ["", "## Segments (strongest first)", ""]
    for segment in report.segments:
        effect = f"{segment.cramers_v:.3f}" if segment.cramers_v is not None else "n/a"
        p_value = f"{segment.p_value:.3g}" if segment.p_value is not None else "n/a"
        lines += [
            f"### {humanize(segment.column)} (effect {effect}, p = {p_value})",
            "",
            _markdown_table(segment.table),
            "",
        ]

    lines += ["## Numeric features (strongest first)", ""]
    rows = []
    for feature in report.features:
        rows.append({
            "Feature": feature.column,
            "AUC" if report.kind == "binary" else "Spearman": round(
                feature.auc if feature.auc is not None else feature.correlation, 3
            ),
            "Direction": feature.direction,
            "Mean (positive)": _round(feature.mean_positive),
            "Mean (negative)": _round(feature.mean_negative),
        })
    lines += [_markdown_table(pd.DataFrame(rows)) if rows else "None.", ""]

    if report.excluded:
        lines += ["## Not used as features", ""]
        for column, reason in report.excluded:
            lines.append(f"- {column}: {reason}")

    return "\n".join(lines) + "\n"


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, 3)


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame.itertuples(index=False):
        lines.append("| " + " | ".join("" if value is None else str(value) for value in row) + " |")
    return "\n".join(lines)
