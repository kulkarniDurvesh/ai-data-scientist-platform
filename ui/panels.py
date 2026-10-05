"""Content builders for each dashboard tab."""

from __future__ import annotations

from typing import Any

import pandas as pd
from dash import dcc, html

from core.schema_inference import format_number, humanize, key_entity_name
import plotly.graph_objects as go

import numpy as np

from core.forecast import DEFAULT_HORIZON, ROW_COUNT, TOTAL, default_aggregation, suggest_freq
from core.modeling import GOAL_TYPES, list_models, target_options
from core.target_analysis import eligible_targets
from visualization.chart_renderer import SERIES_COLORS, apply_theme
from visualization.chart_schema import SUPPORTED_AGGREGATIONS, ChartSpec

from . import components as ui
from .state import DatasetBundle
from .suggestions import suggest_questions


ROLE_LABELS = {
    "measure": "Measure",
    "dimension": "Category",
    "binary": "Yes/No",
    "time": "Date/time",
    "identifier": "ID / key",
    "text": "Free text",
    "constant": "Constant",
    "empty": "Empty",
}

KIND_LABELS = {
    "count": "whole numbers",
    "continuous": "continuous",
    "ratio": "0–1 ratio",
    "binary": "two values",
    "low": "",
    "high": "many values",
}

CHART_TYPE_OPTIONS = [
    {"label": "Auto (recommended)", "value": "AUTO"},
    {"label": "Bar", "value": "BAR"},
    {"label": "Line", "value": "LINE"},
    {"label": "Scatter", "value": "SCATTER"},
    {"label": "Histogram", "value": "HISTOGRAM"},
    {"label": "Box plot", "value": "BOX"},
    {"label": "Pie", "value": "PIE"},
]

AGGREGATION_OPTIONS = [{"label": "Auto", "value": "AUTO"}] + [
    {"label": label, "value": value}
    for value, label in [
        ("SUM", "Sum"),
        ("AVERAGE", "Average"),
        ("COUNT", "Count rows"),
        ("COUNT_DISTINCT", "Count distinct"),
        ("MIN", "Minimum"),
        ("MAX", "Maximum"),
        ("NONE", "None (raw values)"),
    ]
    if value in SUPPORTED_AGGREGATIONS
]

TIME_GRAIN_OPTIONS = [
    {"label": "Auto", "value": "AUTO"},
    {"label": "Day", "value": "D"},
    {"label": "Week", "value": "W"},
    {"label": "Month", "value": "M"},
    {"label": "Quarter", "value": "Q"},
    {"label": "Year", "value": "Y"},
]


# ----------------------------------------------------------------------
# Overview
# ----------------------------------------------------------------------

def overview_panel(bundle: DatasetBundle) -> html.Div:
    df = bundle.df
    schema = bundle.schema
    issues = bundle.quality()

    missing_cells = float(df.isna().mean().mean()) if df.size else 0.0
    duplicates = int(df.duplicated().sum())
    serious = sum(issue["severity"] in {"critical", "warning"} for issue in issues)

    role_counts = pd.Series(
        [column.role for column in schema.columns.values()]
    ).value_counts()

    role_summary = " · ".join(
        f"{count} {ROLE_LABELS.get(role, role).lower()}"
        for role, count in role_counts.items()
    )

    kpis = html.Div(
        [
            ui.kpi_tile("Rows", format_number(len(df))),
            ui.kpi_tile("Columns", format_number(df.shape[1]), role_summary),
            ui.kpi_tile("Missing cells", f"{missing_cells:.1%}"),
            ui.kpi_tile("Duplicate rows", format_number(duplicates)),
            ui.kpi_tile(
                "Quality issues",
                format_number(serious),
                f"{len(issues)} findings in total",
            ),
            ui.kpi_tile(
                "Time axis",
                humanize(schema.primary_time) if schema.primary_time else "None",
                _time_span(df, schema.primary_time),
            ),
        ],
        className="kpi-row",
    )

    quality = (
        html.Div([ui.issue_row(issue) for issue in issues], className="issues")
        if issues
        else ui.message("No data-quality issues found.", "success")
    )

    return html.Div(
        [
            kpis,
            html.Div(
                [
                    ui.section(
                        "Data quality",
                        quality,
                        "Automatic checks for missing values, impossible "
                        "values, outliers, skew, label inconsistencies and "
                        "imbalance.",
                    ),
                    ui.section(
                        "Columns",
                        ui.data_table(
                            _column_table(bundle),
                            page_size=min(max(df.shape[1], 10), 50),
                        ),
                        "How each column was understood. Roles drive the "
                        "automatic charts, the chart builder and the Ask tab.",
                    ),
                ],
                className="grid-2",
            ),
            ui.section(
                "Data preview",
                ui.data_table(df.head(200), page_size=10),
                f"First {min(200, len(df))} rows, all {df.shape[1]} columns "
                f"(scroll sideways to see more).",
            ),
        ]
    )


def _column_table(bundle: DatasetBundle) -> pd.DataFrame:
    rows = []

    for column in bundle.schema.columns.values():
        detail = KIND_LABELS.get(column.kind or "", "")

        if column.name == bundle.schema.primary_time:
            detail = "primary time axis"

        rows.append({
            "Column": column.name,
            "Role": ROLE_LABELS.get(column.role, column.role),
            "Detail": detail,
            "Unique": column.n_unique,
            "Missing %": round(column.missing_ratio * 100, 1),
            "Examples": ", ".join(str(value) for value in column.sample_values),
        })

    return pd.DataFrame(rows)


def _time_span(df: pd.DataFrame, column: str | None) -> str | None:
    if not column:
        return None

    values = pd.to_datetime(df[column], errors="coerce").dropna()

    if values.empty:
        return None

    return f"{values.min():%d %b %Y} – {values.max():%d %b %Y}"


# ----------------------------------------------------------------------
# Automatic charts + insights
# ----------------------------------------------------------------------

def auto_panel(bundle: DatasetBundle) -> html.Div:
    chart_cards = []

    for recommendation in bundle.recommendations():
        spec = recommendation.chart_spec

        try:
            rendered = bundle.render(spec)
            chart_cards.append(
                ui.chart_card(
                    rendered,
                    explanation=recommendation.explanation,
                    badge=(recommendation.category or "").title() or None,
                )
            )
        except (ValueError, TypeError, KeyError) as error:
            chart_cards.append(ui.chart_error_card(spec.title, str(error)))

    charts = (
        html.Div(chart_cards, className="chart-grid")
        if chart_cards
        else ui.message(
            "No automatic charts could be recommended for this dataset.",
        )
    )

    insights = bundle.insights()

    findings = (
        html.Div(
            [_insight_card(index, insight) for index, insight in enumerate(insights, 1)],
            className="insight-list",
        )
        if insights
        else ui.message("No statistically notable patterns were found.")
    )

    return html.Div(
        [
            ui.section(
                "Recommended charts",
                charts,
                "Chosen automatically: each chart is ranked by how much it "
                "reveals (group differences, trends, correlations, skew), "
                "with near-duplicates removed.",
            ),
            ui.section(
                "Key findings",
                findings,
                "Patterns detected by the insight engine, ranked by "
                "interestingness.",
            ),
        ]
    )


def _insight_card(index: int, insight: dict[str, Any]) -> html.Div:
    pattern_type = (insight.get("pattern_type") or "").replace("_", " ")
    score = insight.get("interestingness_score")

    meta = [html.Span(pattern_type.title(), className="badge")]

    if score is not None:
        meta.append(html.Span(f"Score {score:.0f}", className="insight-score"))

    actions = []

    if insight.get("chart_key"):
        actions.append(
            html.Button(
                "Pin chart",
                id=ui.pin_id(insight["chart_key"]),
                n_clicks=0,
                className="btn btn-ghost",
            )
        )

    return html.Div(
        [
            html.Div(
                [html.Span(f"{index}", className="insight-index"), *meta, *actions],
                className="insight-meta",
            ),
            html.P(insight.get("description", ""), className="insight-title"),
            html.P(insight.get("explanation", ""), className="insight-body"),
        ],
        className="card insight-card",
    )


# ----------------------------------------------------------------------
# Chart builder
# ----------------------------------------------------------------------

def builder_panel() -> html.Div:
    def field(label: str, control: Any, hint: str | None = None) -> html.Div:
        children = [html.Label(label, className="field-label"), control]
        if hint:
            children.append(html.Div(hint, className="field-hint"))
        return html.Div(children, className="field")

    controls = html.Div(
        [
            field(
                "Chart type",
                dcc.Dropdown(
                    id="b-type",
                    options=CHART_TYPE_OPTIONS,
                    value="AUTO",
                    clearable=False,
                ),
            ),
            field("X axis", dcc.Dropdown(id="b-x", placeholder="Choose a column")),
            field("Y axis / value", dcc.Dropdown(id="b-y", placeholder="Optional")),
            field(
                "Aggregation",
                dcc.Dropdown(
                    id="b-agg",
                    options=AGGREGATION_OPTIONS,
                    value="AUTO",
                    clearable=False,
                ),
            ),
            field("Color / group by", dcc.Dropdown(id="b-color", placeholder="None")),
            field(
                "Time grain",
                dcc.Dropdown(
                    id="b-grain",
                    options=TIME_GRAIN_OPTIONS,
                    value="AUTO",
                    clearable=False,
                ),
                "Used when X is a date.",
            ),
            field(
                "Top N",
                dcc.Input(
                    id="b-limit",
                    type="number",
                    min=1,
                    step=1,
                    placeholder="All",
                    className="input",
                ),
            ),
        ],
        className="controls",
    )

    filters = html.Div(
        [
            field("Filter column", dcc.Dropdown(id="b-filter-col", placeholder="No filter")),
            html.Div(
                field(
                    "Keep values",
                    dcc.Dropdown(id="b-filter-values", multi=True, placeholder="All values"),
                ),
                id="b-filter-values-wrap",
            ),
            html.Div(
                [
                    field("Min", dcc.Input(id="b-filter-min", type="number", className="input")),
                    field("Max", dcc.Input(id="b-filter-max", type="number", className="input")),
                ],
                id="b-filter-range-wrap",
                className="range-fields",
                style={"display": "none"},
            ),
        ],
        className="controls controls-filters",
    )

    return html.Div(
        [
            ui.section(
                "Build a chart",
                html.Div([controls, filters], className="card"),
                "Pick columns and the chart updates immediately. Leave chart "
                "type and aggregation on Auto to let the schema decide.",
            ),
            dcc.Loading(html.Div(id="b-output"), type="dot", color="var(--accent)"),
        ]
    )


def column_options(
    bundle: DatasetBundle,
    roles: set[str] | None = None,
    include_high_cardinality: bool = True,
) -> list[dict[str, str]]:

    usable = {"measure", "dimension", "binary", "time", "identifier"}
    roles = roles or usable

    options = []

    for column in bundle.schema.columns.values():
        if column.role not in roles:
            continue
        if column.high_cardinality and not include_high_cardinality:
            continue

        role = ROLE_LABELS.get(column.role, column.role)
        options.append({"label": f"{column.name}  ·  {role}", "value": column.name})

    return options


# ----------------------------------------------------------------------
# Ask
# ----------------------------------------------------------------------

def ask_panel(bundle: DatasetBundle) -> html.Div:
    suggestions = suggest_questions(bundle.df, bundle.schema)

    return html.Div(
        [
            ui.section(
                "Ask a question",
                html.Div(
                    [
                        html.Div(
                            [
                                dcc.Input(
                                    id="ask-input",
                                    type="text",
                                    placeholder="e.g. " + (suggestions[0] if suggestions else "What is the average ...?"),
                                    debounce=False,
                                    n_submit=0,
                                    className="input ask-input",
                                ),
                                html.Button("Ask", id="ask-button", n_clicks=0, className="btn btn-primary"),
                            ],
                            className="ask-row",
                        ),
                        html.Div(
                            [html.Span("Try:", className="suggest-label")]
                            + [
                                html.Button(
                                    question,
                                    id={"type": "suggest", "index": index},
                                    n_clicks=0,
                                    className="chip",
                                )
                                for index, question in enumerate(suggestions)
                            ],
                            className="suggestions",
                        ),
                    ],
                    className="card",
                ),
                "Questions are answered by computing on the data: lists "
                "(\"which ... are in ...\"), counts, sums, averages, min/max, "
                "top/bottom N, grouping (\"by\"), filters, dates (\"in March "
                "2025\") and charts (\"pie chart of ... by ...\").",
            ),
            dcc.Loading(
                html.Div(answer_cards(bundle), id="ask-output"),
                type="dot",
                color="var(--accent)",
            ),
        ]
    )


def answer_cards(bundle: DatasetBundle) -> list[html.Div]:
    cards = []

    for index, entry in enumerate(bundle.ask_history):
        children = [html.Div(entry["question"], className="answer-question")]

        if entry.get("notice"):
            children.append(ui.message(entry["notice"], "info"))
        elif entry["error"]:
            children.append(
                ui.message(f"I couldn't answer that: {entry['error']}", "error")
            )
        else:
            children.append(html.Div(entry["answer"], className="answer-text"))

            table = entry.get("table")
            if isinstance(table, pd.DataFrame) and not table.empty:
                children.append(ui.data_table(table, page_size=8))

            if entry.get("chart_key"):
                try:
                    rendered = bundle.render(entry["chart_key"])
                    children.append(ui.chart_card(rendered))
                except (ValueError, TypeError, KeyError):
                    pass

        cards.append(
            html.Div(
                children,
                className="card answer-card" + (" answer-latest" if index == 0 else ""),
            )
        )

    return cards


# ----------------------------------------------------------------------
# Board
# ----------------------------------------------------------------------

def board_panel(bundle: DatasetBundle | None, keys: list[str]) -> html.Div:
    if bundle is None:
        return html.Div()

    if not keys:
        return ui.empty_state(
            "Your board is empty",
            "Use “Pin to board” on any automatic chart, finding, built "
            "chart or answer to collect it here.",
        )

    cards = []

    for key in keys:
        if key not in bundle.charts:
            continue
        try:
            cards.append(ui.chart_card(bundle.render(key), action="unpin"))
        except (ValueError, TypeError, KeyError) as error:
            cards.append(ui.chart_error_card(bundle.charts[key].title, str(error)))

    return html.Div(
        [
            html.Div(
                html.Button("Clear board", id="board-clear", n_clicks=0, className="btn btn-ghost"),
                className="board-toolbar",
            ),
            html.Div(cards, className="chart-grid"),
        ]
    )


# ----------------------------------------------------------------------
# Target (training-table EDA)
# ----------------------------------------------------------------------

# Segment charts shown above the full segment table.
TARGET_SEGMENT_CHARTS = 4


def target_panel(bundle: DatasetBundle) -> html.Div:
    candidates = bundle.target_candidates()
    eligible = eligible_targets(bundle.df, bundle.schema)

    if not eligible:
        return ui.empty_state(
            "No target to analyse",
            "Target analysis needs a two-valued column (e.g. 0/1, Yes/No) "
            "or a numeric column to predict.",
        )

    suggested = [candidate.column for candidate in candidates]
    ordered = suggested + [column for column in eligible if column not in suggested]

    options = [
        {
            "label": f"{column}  (suggested)" if column in suggested[:1] else column,
            "value": column,
        }
        for column in ordered
    ]

    default = ordered[0]

    picker = html.Div(
        [
            html.Div(
                [
                    html.Label("Target column", className="field-label"),
                    dcc.Dropdown(
                        id="target-select",
                        options=options,
                        value=default,
                        clearable=False,
                        style={"minWidth": "260px"},
                    ),
                ],
                className="field",
            ),
            html.Button(
                "Download report",
                id="target-download-button",
                n_clicks=0,
                className="btn",
            ),
            dcc.Download(id="target-download"),
        ],
        className="target-picker",
    )

    intro = (
        "Explore a training table against the column a model should "
        "predict: how often the outcome happens, which segments and "
        "features relate to it, leakage risks and how to split the data."
    )
    if candidates:
        intro += f" Suggested target: {candidates[0].column} ({candidates[0].reason})."

    return html.Div(
        [
            ui.section("Target analysis", picker, intro),
            dcc.Loading(
                html.Div(target_body(bundle, default), id="target-body"),
                type="dot",
                color="var(--accent)",
            ),
        ]
    )


def target_body(bundle: DatasetBundle, target: str | None) -> html.Div:
    if not target:
        return html.Div()

    try:
        report = bundle.target_report(target)
    except ValueError as error:
        return ui.message(str(error), "error")

    rate_title = "Target rate" if report.kind == "binary" else "Average target"

    feature_note = (
        "How well each column separates the outcome on its own. AUC 0.5 "
        "means no signal; the further from 0.5, the stronger. Values "
        "beyond 0.98 (or below 0.02) usually mean leakage."
        if report.kind == "binary"
        else "Rank correlation (Spearman) with the target."
    )

    return html.Div(
        [
            _target_kpis(report),
            ui.section(
                "Warnings and checks",
                _target_findings(report),
                "Leakage, labels, imbalance and drift. Critical items "
                "should be fixed before training.",
            ),
            ui.section("Train / test split", _target_split(report)),
            ui.section(
                f"{rate_title} by segment",
                _target_segments(bundle, report),
                "Every category column, strongest relationship first.",
            ),
            _target_over_time(bundle, report),
            ui.section("Numeric features", _target_features(report), feature_note),
            ui.section("Not used as features", _target_excluded(report)),
        ]
    )


def _target_kpis(report) -> html.Div:
    tiles = [
        ui.kpi_tile("Target", humanize(report.target)),
        ui.kpi_tile(
            "Rows with a target",
            format_number(report.n_labelled),
            f"of {format_number(report.n_rows)}",
        ),
    ]

    if report.kind == "binary":
        tiles.append(
            ui.kpi_tile(
                "Positive rate",
                f"{report.positive_rate:.1%}",
                f"positive = {report.positive_label}",
            )
        )
        if report.imbalance is not None:
            tiles.append(ui.kpi_tile("Balance", report.imbalance.title))
    else:
        tiles.append(ui.kpi_tile("Average", format_number(report.mean)))

    if report.panel is not None:
        panel = report.panel
        tiles.append(
            ui.kpi_tile(
                "Panel data",
                f"{format_number(panel.n_entities)} × {format_number(panel.n_periods)}",
                f"{humanize(panel.entity)} × {humanize(panel.time)} periods",
            )
        )

    return html.Div(tiles, className="kpi-row")


def _target_findings(report) -> html.Div:
    if not report.findings:
        return ui.message("No leakage, label or imbalance problems found.", "success")

    return html.Div(
        [
            ui.issue_row({
                "severity": finding.severity,
                "message": f"{finding.title}. {finding.detail}",
            })
            for finding in report.findings
        ],
        className="issues",
    )


def _target_split(report) -> html.Div:
    split = report.split
    children = [html.P(split.summary, className="card-explanation")]

    if split.table is not None:
        children.append(ui.data_table(split.table, page_size=4))

    return html.Div(children, className="card")


def _target_segments(bundle: DatasetBundle, report) -> html.Div:
    if not report.segments:
        return ui.message("No category columns to compare.")

    charts = []

    # Charts average the target column, so it must be stored as numbers.
    if bundle.schema.is_numeric(report.target):
        for segment in report.segments[:TARGET_SEGMENT_CHARTS]:
            spec = ChartSpec(
                chart_type="BAR",
                x_column=segment.column,
                y_column=report.target,
                aggregation="AVERAGE",
                sort="DESC",
            )
            try:
                charts.append(
                    ui.chart_card(
                        bundle.render(spec),
                        explanation=_segment_text(report, segment),
                    )
                )
            except (ValueError, TypeError, KeyError) as error:
                charts.append(ui.chart_error_card(spec.title, str(error)))

    summary = pd.DataFrame([
        {
            "Column": segment.column,
            "Effect": None if segment.cramers_v is None else round(segment.cramers_v, 3),
            "p-value": None if segment.p_value is None else float(f"{segment.p_value:.3g}"),
            "Highest": f"{segment.highest[0]} ({_target_value(report, segment.highest[1])})",
            "Lowest": f"{segment.lowest[0]} ({_target_value(report, segment.lowest[1])})",
            "Groups": len(segment.table),
        }
        for segment in report.segments
    ])

    children = []
    if charts:
        children.append(html.Div(charts, className="chart-grid"))
    children.append(ui.data_table(summary, page_size=10))

    return html.Div(children)


def _segment_text(report, segment) -> str:
    high, low = segment.highest, segment.lowest
    text = (
        f"{high[0]} is highest ({_target_value(report, high[1])}), "
        f"{low[0]} lowest ({_target_value(report, low[1])})"
    )
    if segment.p_value is not None:
        text += (
            "; the difference is statistically significant."
            if segment.p_value < 0.05
            else "; the difference could be chance (p ≥ 0.05)."
        )
    return text


def _target_value(report, value: float) -> str:
    return f"{value:.1%}" if report.kind == "binary" else format_number(value)


def _target_over_time(bundle: DatasetBundle, report) -> html.Div:
    if report.over_time is None or not bundle.schema.is_numeric(report.target):
        return html.Div()

    spec = ChartSpec(
        chart_type="LINE",
        x_column=report.time_column,
        y_column=report.target,
        aggregation="AVERAGE",
        time_grain=report.time_grain or "AUTO",
    )

    try:
        card = ui.chart_card(
            bundle.render(spec),
            explanation="Watch for drift: a model trained on early periods "
            "is scored on later ones.",
        )
    except (ValueError, TypeError, KeyError) as error:
        card = ui.chart_error_card(spec.title, str(error))

    return ui.section("Target over time", html.Div(card, className="chart-grid"))


def _target_features(report) -> html.Div:
    if not report.features:
        return ui.message("No numeric columns to compare.")

    rows = []
    for feature in report.features:
        row = {"Feature": feature.column}
        if report.kind == "binary":
            row["AUC"] = round(feature.auc, 3)
            row["Signal"] = _signal_label(feature.strength)
            row["Direction"] = feature.direction
            row[f"Mean when {report.positive_label}"] = round(feature.mean_positive, 3)
            row["Mean otherwise"] = round(feature.mean_negative, 3)
        else:
            row["Spearman"] = round(feature.correlation, 3)
            row["Signal"] = _signal_label(feature.strength)
            row["Direction"] = feature.direction
        rows.append(row)

    return ui.data_table(pd.DataFrame(rows), page_size=12)


def _signal_label(strength: float) -> str:
    if strength >= 0.96:
        return "Suspicious (leak?)"
    if strength >= 0.4:
        return "Strong"
    if strength >= 0.2:
        return "Moderate"
    if strength >= 0.05:
        return "Weak"
    return "None"


def _target_excluded(report) -> html.Div:
    if not report.excluded:
        return ui.message("Every column can be used as a feature.", "success")

    table = pd.DataFrame(report.excluded, columns=["Column", "Why"])
    return ui.data_table(table, page_size=10)


# ----------------------------------------------------------------------
# Build model (goal-driven model builder)
# ----------------------------------------------------------------------

# Rows of the scored list shown in the dashboard (the download has all).
MODEL_SCORED_ROWS = 200


def model_panel(bundle: DatasetBundle) -> html.Div:
    options = {
        goal: target_options(bundle.df, bundle.schema, goal)
        for goal in GOAL_TYPES
    }
    available = [goal for goal in ("rank", "classify", "regress") if options[goal]]

    if not available:
        return ui.empty_state(
            "No column to predict",
            "Model building needs a two-valued column (yes/no) or a numeric "
            "column as the target.",
        )

    goal = available[0]

    goal_picker = dcc.RadioItems(
        id="model-goal",
        options=[
            {
                "label": html.Span(
                    [
                        html.Strong(GOAL_TYPES[key]["label"]),
                        html.Span(f" — {GOAL_TYPES[key]['description']}", className="goal-description"),
                    ]
                ),
                "value": key,
                "disabled": not options[key],
            }
            for key in ("rank", "classify", "regress")
        ],
        value=goal,
        className="goal-options",
        labelClassName="goal-option",
    )

    controls = html.Div(
        [
            html.Div(
                [
                    html.Label("What to predict", className="field-label"),
                    dcc.Dropdown(
                        id="model-target",
                        options=_target_dropdown(options[goal]),
                        value=options[goal][0],
                        clearable=False,
                        style={"minWidth": "260px"},
                    ),
                ],
                className="field",
            ),
            html.Button("Train models", id="model-train", n_clicks=0, className="btn btn-primary"),
        ],
        className="target-picker",
    )

    return html.Div(
        [
            dcc.Store(id="model-job"),
            dcc.Interval(id="model-poll", interval=1000, disabled=True),
            ui.section(
                "What do you want to build?",
                html.Div([goal_picker, controls], className="card"),
                "Pick a goal and the column to predict. The platform proposes "
                "the features, excludes IDs, dates and leaky columns, splits "
                "the data honestly, compares several models against a "
                "baseline, explains the best one and scores the latest rows.",
            ),
            html.Div(model_setup_view(bundle, goal, options[goal][0]), id="model-setup"),
            # The last trained model stays visible after a page refresh.
            html.Div(
                model_results_view(bundle.latest_model) if bundle.latest_model else None,
                id="model-results",
            ),
            ui.section("Saved models", html.Div(saved_models_view(), id="model-saved")),
        ]
    )


def model_target_options(bundle: DatasetBundle, goal: str) -> tuple[list[dict], str | None]:
    columns = target_options(bundle.df, bundle.schema, goal)
    return _target_dropdown(columns), (columns[0] if columns else None)


def _target_dropdown(columns: list[str]) -> list[dict[str, str]]:
    return [
        {"label": f"{column}  (suggested)" if index == 0 else column, "value": column}
        for index, column in enumerate(columns)
    ]


def model_setup_view(bundle: DatasetBundle, goal: str | None, target: str | None) -> html.Div:
    if not goal or not target:
        return html.Div()

    try:
        spec = bundle.model_goal(goal, target)
    except ValueError as error:
        return ui.message(str(error), "error")

    if spec.time_column:
        split = (
            f"By time on {spec.time_column}: earlier periods train the models, "
            f"the latest ~20% of periods test them."
        )
    elif spec.task == "classification":
        split = "Stratified random split (no date column), keeping the positive rate in every part."
    else:
        split = "Random split (no date column)."

    rows = [
        {"Setting": "Task", "Value": "Classification" if spec.task == "classification" else "Regression"},
        {"Setting": "Target", "Value": spec.target},
        {"Setting": "Features used", "Value": f"{len(spec.features)}: " + ", ".join(spec.features)},
        {"Setting": "Split", "Value": split},
        {"Setting": "Entity (names rows)", "Value": spec.entity or "row number"},
        {
            "Setting": "Models compared",
            "Value": "Baseline, logistic regression, random forest, gradient boosting"
            if spec.task == "classification"
            else "Baseline, ridge regression, random forest, gradient boosting",
        },
    ]

    children = [ui.data_table(pd.DataFrame(rows), page_size=10)]

    if spec.excluded:
        children.append(html.H3("Left out of the model", className="card-title"))
        children.append(
            ui.data_table(pd.DataFrame(spec.excluded, columns=["Column", "Why"]), page_size=8)
        )

    return ui.section("Proposed setup", html.Div(children, className="card"))


def model_progress_view(job: dict) -> html.Div:
    steps = job.get("messages") or ["Starting"]
    return html.Div(
        [
            ui.message("Training in progress — the page stays usable.", "info"),
            html.Ul([html.Li(step) for step in steps], className="progress-steps"),
        ],
        className="card",
    )


def model_results_view(result) -> html.Div:
    task = result.task
    best, base = result.test_metrics, result.baseline_metrics

    if task == "classification":
        share = int(best.get("top_share", 0.2) * 100)
        tiles = [
            ui.kpi_tile("Chosen model", result.best_name, f"trained in {result.seconds:.0f}s"),
            ui.kpi_tile("Test PR-AUC", _fmt(best.get("pr_auc")), f"baseline {_fmt(base.get('pr_auc'))}"),
            ui.kpi_tile("Test ROC-AUC", _fmt(best.get("roc_auc")), "0.5 = random"),
            ui.kpi_tile(f"Top {share}% capture", _pct(best.get("recall_at_top")), f"random list: {share}%"),
            ui.kpi_tile(f"Top {share}% lift", _times(best.get("lift_at_top")), "vs average rate"),
        ]
    else:
        improvement = 1 - best["mae"] / base["mae"] if base.get("mae") else float("nan")
        tiles = [
            ui.kpi_tile("Chosen model", result.best_name, f"trained in {result.seconds:.0f}s"),
            ui.kpi_tile("Test MAE", format_number(best["mae"]), f"baseline {format_number(base['mae'])}"),
            ui.kpi_tile("Improvement", _pct(improvement), "lower error than baseline"),
            ui.kpi_tile("Test R²", _fmt(best.get("r2")), "1 = perfect"),
        ]

    heading = html.H2(
        f"Results: {GOAL_TYPES[result.spec.goal_type]['label']} — {result.spec.target}",
        className="section-title",
    )
    notes = [ui.message(result.summary, "success")]
    notes += [ui.message(text, "error") for text in result.warnings]

    charts = [_importance_card(result)]
    if result.lift is not None:
        charts.insert(0, _gains_card(result))

    scored = result.scored.head(MODEL_SCORED_ROWS)

    sections = [
        heading,
        html.Div(tiles, className="kpi-row"),
        html.Div(notes),
        ui.section(
            "Model comparison",
            html.Div(
                [
                    html.P(result.split.description, className="card-explanation"),
                    ui.data_table(result.candidates, page_size=6),
                ],
                className="card",
            ),
            "The chosen model is picked on the validation window; every "
            "model is then scored once on the test period for comparison.",
        ),
        ui.section("How well it ranks and what drives it", html.Div(charts, className="chart-grid")),
    ]

    if result.lift is not None:
        sections.append(
            ui.section(
                "Lift by group (test period)",
                ui.data_table(result.lift, page_size=10),
                "Test rows ranked by predicted probability and cut into ten "
                "equal groups.",
            )
        )

    sections.append(
        ui.section(
            "Predictions",
            html.Div(
                [
                    html.P(
                        f"Scored: {result.scored_description}. Reasons name the "
                        f"most important columns where the row is unusual.",
                        className="card-explanation",
                    ),
                    html.Div(
                        [
                            html.Button("Download all predictions (CSV)", id="model-download-button", n_clicks=0, className="btn"),
                            html.Button("Save model", id="model-save-button", n_clicks=0, className="btn"),
                            dcc.Download(id="model-download"),
                            html.Span(id="model-save-message", className="save-message"),
                        ],
                        className="target-picker",
                    ),
                    ui.data_table(scored, page_size=10),
                ],
                className="card",
            ),
        )
    )

    return html.Div(sections)


def _gains_card(result) -> html.Div:
    lift = result.lift
    x = [0] + list(lift["Top % of list"])
    y = [0] + list(lift["Cumulative capture"] * 100)

    figure = go.Figure()
    figure.add_trace(go.Scatter(x=x, y=y, mode="lines+markers", name=result.best_name, line={"color": SERIES_COLORS[0], "width": 3}))
    figure.add_trace(go.Scatter(x=[0, 100], y=[0, 100], mode="lines", name="Random list", line={"color": SERIES_COLORS[1], "dash": "dash"}))
    apply_theme(figure, height=360)
    figure.update_xaxes(title_text="Top % of ranked list", ticksuffix="%")
    figure.update_yaxes(title_text="% of positives captured", ticksuffix="%", range=[0, 105])

    return ui.figure_card(
        "Cumulative gains (test period)",
        figure,
        "How many of the actual positives are found when you work down the "
        "ranked list. The further above the dashed line, the better.",
    )


def _importance_card(result) -> html.Div:
    table = result.importance.head(12).iloc[::-1]

    figure = go.Figure(
        go.Bar(
            x=table["Importance"],
            y=[humanize(feature) for feature in table["Feature"]],
            orientation="h",
            marker_color=SERIES_COLORS[0],
            error_x={"type": "data", "array": table["Spread"], "color": "#9a9894"},
        )
    )
    apply_theme(figure, height=360)
    figure.update_xaxes(title_text="Score drop when the column is shuffled")

    return ui.figure_card(
        "What drives the model",
        figure,
        "Permutation importance on the test period: how much the model gets "
        "worse when one column's values are shuffled.",
    )


def saved_models_view() -> html.Div:
    saved = list_models()

    if not saved:
        return ui.message("No saved models yet. Train a model and click Save model.")

    rows = []
    for meta in saved:
        metrics = meta.get("test_metrics", {})
        key = (
            f"PR-AUC {metrics['pr_auc']:.3f}" if "pr_auc" in metrics
            else f"MAE {metrics['mae']:.3g}" if "mae" in metrics
            else ""
        )
        rows.append({
            "Name": meta.get("name"),
            "Created (UTC)": meta.get("created"),
            "Dataset": meta.get("dataset"),
            "Target": meta.get("goal", {}).get("target"),
            "Model": meta.get("best_model"),
            "Test metric": key,
        })

    return ui.data_table(pd.DataFrame(rows), page_size=8)


def _fmt(value) -> str:
    return "–" if value is None or pd.isna(value) else f"{value:.3f}"


def _pct(value) -> str:
    return "–" if value is None or pd.isna(value) else f"{value:.0%}"


def _times(value) -> str:
    return "–" if value is None or pd.isna(value) else f"{value:.1f}×"


# ----------------------------------------------------------------------
# Recommend (next-best-contact plans)
# ----------------------------------------------------------------------

NO_GROUP_VALUE = "__none__"
CURRENT_SHEET = "__current__"
PLAN_PERIODS = [
    {"label": "Next 5 working days", "value": "5"},
    {"label": "Next 10 working days", "value": "10"},
    {"label": "Next calendar month", "value": "month"},
    {"label": "Custom number of days", "value": "custom"},
]


def sheet_from_value(value: str | None) -> str | None:
    return None if value in (None, CURRENT_SHEET) else value


def recommend_panel(bundle: DatasetBundle) -> html.Div:
    sheets = bundle.interaction_sheets()

    if not sheets:
        return ui.empty_state(
            "No interaction table found",
            "Recommendations need a table of interactions: two repeated keys "
            "(who contacted what), a date and an outcome column.",
        )

    sheet = sheets[0]
    sheet_picker = [
        html.Div(
            [
                html.Label("Interaction table", className="field-label"),
                dcc.Dropdown(
                    id="rec-sheet",
                    options=[
                        {"label": name if name else bundle.name, "value": name or CURRENT_SHEET}
                        for name in sheets
                    ],
                    value=sheet or CURRENT_SHEET,
                    clearable=False,
                    style={"minWidth": "200px"},
                ),
            ],
            className="field",
        )
    ]

    intro = (
        "Learns which contacts tend to succeed from the interaction history "
        "(using only information available before each contact), then plans "
        "each user's next working days within their group, respecting a daily "
        "capacity and a minimum gap between contacts with the same item. A "
        "backtest compares the model with simple rules on held-out weeks."
    )
    if sheet != bundle.sheet:
        intro += f" Using the '{sheet}' sheet, which holds the interactions."

    return html.Div(
        [
            dcc.Store(id="rec-job"),
            dcc.Interval(id="rec-poll", interval=1000, disabled=True),
            ui.section(
                "Recommend next contacts",
                html.Div(
                    [
                        html.Div(sheet_picker, className="target-picker"),
                        html.Div(recommend_roles_view(bundle, sheet), id="rec-roles"),
                    ],
                    className="card",
                ),
                intro,
            ),
            html.Div(
                recommend_results_view(bundle.latest_recommendation) if bundle.latest_recommendation else None,
                id="rec-results",
            ),
        ]
    )


def _dropdown(component_id: str, label: str, options: list, value, multi: bool = False, width: str = "200px") -> html.Div:
    return html.Div(
        [
            html.Label(label, className="field-label"),
            dcc.Dropdown(
                id=component_id,
                options=options,
                value=value,
                multi=multi,
                clearable=multi,
                style={"minWidth": width},
            ),
        ],
        className="field",
    )


def _number(component_id: str, label: str, value, placeholder: str) -> html.Div:
    return html.Div(
        [
            html.Label(label, className="field-label"),
            dcc.Input(id=component_id, type="number", min=1, step=1, value=value,
                      placeholder=placeholder, className="input", style={"width": "120px"}),
        ],
        className="field",
    )


def recommend_roles_view(bundle: DatasetBundle, sheet: str | None) -> html.Div:
    try:
        options, roles = bundle.interaction_options(sheet)
    except ValueError as error:
        return ui.message(str(error), "error")

    frame, _, _, _ = bundle.interaction_context(sheet)
    as_options = lambda columns: [{"label": column, "value": column} for column in columns]

    groups = bundle.interaction_groups(sheet, roles.user, roles.item)

    return html.Div(
        [
            html.Div(
                [
                    _dropdown("rec-user", "Who acts (user)", as_options(options.users), roles.user),
                    _dropdown("rec-item", "Contacted (item)", as_options(options.items), roles.item),
                    _dropdown("rec-time", "When", as_options(options.times), roles.time),
                    _dropdown("rec-outcome", "Outcome", as_options(options.outcomes), roles.outcome),
                ],
                className="target-picker",
            ),
            html.Div(
                [
                    _dropdown(
                        "rec-success", "Counts as success",
                        success_options(frame, roles.outcome), roles.success_values,
                        multi=True, width="320px",
                    ),
                    _dropdown("rec-group", "Plan within", group_dropdown(groups), _group_value(groups, roles), width="220px"),
                ],
                className="target-picker",
            ),
            html.Div(
                [
                    _dropdown("rec-period", "Plan for", PLAN_PERIODS, "month", width="220px"),
                    _number("rec-days", "Working days (custom)", 22, "22"),
                    html.Div(
                        [
                            html.Label("Start from (optional)", className="field-label"),
                            dcc.DatePickerSingle(
                                id="rec-start",
                                placeholder="after last contact",
                                display_format="YYYY-MM-DD",
                                clearable=True,
                            ),
                        ],
                        className="field",
                    ),
                ],
                className="target-picker",
            ),
            html.Div(
                [
                    _number("rec-capacity", "Max contacts per user per day", None, "auto"),
                    _number("rec-gap", "Min days between contacts", None, "auto"),
                    dcc.Checklist(
                        id="rec-owner",
                        options=[{
                            "label": f" Keep each {entity_noun(roles.item)} with one {entity_noun(roles.user)}",
                            "value": "owner",
                        }],
                        value=["owner"],
                        className="field",
                    ),
                    html.Button("Build plan", id="rec-build", n_clicks=0, className="btn btn-primary"),
                ],
                className="target-picker",
            ),
            html.P(
                "Next calendar month plans every working day of the month after "
                "the last recorded contact (or of the start date's month). Users "
                "only get items of their own group. Blank capacity and gap are "
                "taken from the history; contacts are spread evenly over the days.",
                className="card-explanation",
            ),
        ]
    )


def entity_noun(column: str) -> str:
    """The thing a key column names, from the data: DoctorId -> doctor."""
    return key_entity_name(column) or humanize(column).lower()


def success_options(frame: pd.DataFrame, outcome: str) -> list[dict]:
    values = frame[outcome].dropna().unique() if outcome in frame.columns else []
    return [{"label": str(value), "value": value} for value in values]


def group_dropdown(groups: list[tuple[str, str, str]]) -> list[dict]:
    options = [
        {"label": label if user == item else f"{label} (user's and item's)", "value": label}
        for label, user, item in groups
    ]
    return options + [{"label": "No grouping", "value": NO_GROUP_VALUE}]


def _group_value(groups, roles) -> str:
    for label, user, item in groups:
        if user == roles.user_group and item == roles.item_group:
            return label
    return NO_GROUP_VALUE


def recommend_progress_view(job: dict) -> html.Div:
    return model_progress_view(job)


def recommend_results_view(result) -> html.Div:
    model = result.model
    best = model.test_metrics
    backtest = result.backtest.set_index("Strategy")
    settings = result.settings

    tiles = [
        ui.kpi_tile("Success model", model.best_name, f"PR-AUC {_fmt(best.get('pr_auc'))} · ROC-AUC {_fmt(best.get('roc_auc'))}"),
        ui.kpi_tile("Backtest lift", _times(backtest.loc["Model", "Lift"]), "top 20% vs all contacts"),
        ui.kpi_tile("Success rate (top 20%)", _pct(backtest.loc["Model", "Success rate"]), f"random {_pct(backtest.loc['Random', 'Success rate'])}"),
        ui.kpi_tile("Contacts planned", format_number(len(result.plan)), f"{result.n_users} users · {settings.days} day(s)"),
        ui.kpi_tile("Rules", f"≤ {settings.capacity}/day", f"≥ {settings.min_gap_days} days apart"),
    ]

    notes = [ui.message(result.summary, "success")]
    notes += [ui.message(text, "info") for text in result.warnings]

    users = sorted(result.plan[result.roles.user].unique()) if not result.plan.empty else []

    return html.Div(
        [
            html.H2("Results: next-contact plan", className="section-title"),
            html.Div(tiles, className="kpi-row"),
            html.Div(notes),
            ui.section(
                "Backtest: does prioritising help?",
                html.Div(
                    [_backtest_card(result), _importance_card(model)],
                    className="chart-grid",
                ),
                "In each held-out week, the top 20% of that week's actual "
                "contacts are picked by each strategy; their real success rate "
                "is compared.",
            ),
            ui.section("Backtest table", ui.data_table(result.backtest, page_size=6)),
            ui.section(
                "Plan",
                html.Div(
                    [
                        html.Div(
                            [
                                html.Button("Download plan (CSV)", id="rec-download-button", n_clicks=0, className="btn"),
                                dcc.Download(id="rec-download"),
                            ],
                            className="target-picker",
                        ),
                        html.H3("By group", className="card-title"),
                        ui.data_table(result.summary_table, page_size=12) if not result.summary_table.empty else html.Div(),
                        html.H3("By user", className="card-title"),
                        ui.data_table(result.user_table, page_size=10) if not result.user_table.empty else html.Div(),
                    ],
                    className="card",
                ),
            ),
            ui.section(
                "Day-wise plan per user",
                html.Div(
                    [
                        html.Div(
                            [_dropdown("rec-user-view", "User", [{"label": u, "value": u} for u in users], users[0] if users else None, width="220px")],
                            className="target-picker",
                        ),
                        html.Div(recommend_user_plan(result, users[0]) if users else None, id="rec-user-plan"),
                    ],
                    className="card",
                ),
                "Every working day of the period for one user, with the items to "
                "contact in order, their predicted success and the reasons.",
            ),
            ui.section(
                "Success model comparison",
                html.Div(
                    [
                        html.P(model.split.description, className="card-explanation"),
                        ui.data_table(model.candidates, page_size=6),
                    ],
                    className="card",
                ),
            ),
        ]
    )


def recommend_user_plan(result, user) -> html.Div:
    roles = result.roles
    plan = result.plan[result.plan[roles.user] == user].copy()

    if plan.empty:
        return ui.message("No contacts planned for this user.")

    days = sorted(plan["Day"].unique())
    plan["Day"] = pd.to_datetime(plan["Day"]).dt.strftime("%a %Y-%m-%d")
    drop = [roles.user] + (["Group"] if "Group" in plan.columns else [])
    group = plan["Group"].iloc[0] if "Group" in plan.columns else None

    note = (
        f"{user}{f' ({group})' if group else ''}: {len(plan)} contacts on {len(days)} "
        f"day(s), {plan[roles.item].nunique()} distinct items."
    )

    return html.Div(
        [
            html.P(note, className="card-explanation"),
            ui.data_table(plan.drop(columns=drop), page_size=15),
        ]
    )


def _backtest_card(result) -> html.Div:
    table = result.backtest
    colors = [SERIES_COLORS[0] if name == "Model" else "#b9b7b1" for name in table["Strategy"]]

    figure = go.Figure(
        go.Bar(
            x=table["Strategy"],
            y=table["Success rate"] * 100,
            marker_color=colors,
            text=[f"{value:.0%}" for value in table["Success rate"]],
            textposition="outside",
        )
    )
    apply_theme(figure, height=360)
    figure.update_yaxes(title_text="Success rate of picked contacts", ticksuffix="%", range=[0, 105])

    return ui.figure_card(
        "Success rate of the contacts each strategy picks",
        figure,
        "Higher is better. 'Random' is the success rate without prioritisation.",
    )


# ----------------------------------------------------------------------
# Forecast
# ----------------------------------------------------------------------

FREQ_OPTIONS = [
    {"label": "Day", "value": "D"},
    {"label": "Week", "value": "W"},
    {"label": "Month", "value": "M"},
    {"label": "Quarter", "value": "Q"},
]
AGG_OPTIONS = [
    {"label": "Sum", "value": "sum"},
    {"label": "Average", "value": "mean"},
    {"label": "Distinct count", "value": "nunique"},
]
NO_SPLIT = "__none__"


def forecast_panel(bundle: DatasetBundle) -> html.Div:
    sheets = bundle.forecast_sheets()

    if not sheets:
        return ui.empty_state(
            "Nothing to forecast",
            "Forecasting needs a table with a date column and at least 20 rows.",
        )

    # After a refresh, show the controls of the forecast on screen.
    last = bundle.latest_forecast
    sheet = bundle.latest_forecast_sheet if last and bundle.latest_forecast_sheet in sheets else sheets[0]

    picker = html.Div(
        [
            html.Label("Table", className="field-label"),
            dcc.Dropdown(
                id="fc-sheet",
                options=[{"label": name if name else bundle.name, "value": name or CURRENT_SHEET} for name in sheets],
                value=sheet or CURRENT_SHEET,
                clearable=False,
                style={"minWidth": "200px"},
            ),
        ],
        className="field",
    )

    return html.Div(
        [
            dcc.Store(id="fc-job"),
            dcc.Interval(id="fc-poll", interval=1000, disabled=True),
            ui.section(
                "Forecast",
                html.Div(
                    [
                        html.Div([picker], className="target-picker"),
                        html.Div(
                            forecast_controls_view(bundle, sheet, last.spec if last and sheet == bundle.latest_forecast_sheet else None),
                            id="fc-controls",
                        ),
                    ],
                    className="card",
                ),
                "Forecast a measure over time, in total and per group. Several "
                "models are compared with simple baselines on past periods "
                "(rolling backtest); the best one forecasts the next periods "
                "with approximate 80% and 95% intervals.",
            ),
            html.Div(
                forecast_results_view(bundle.latest_forecast) if bundle.latest_forecast else None,
                id="fc-results",
            ),
        ]
    )


def forecast_controls_view(bundle: DatasetBundle, sheet: str | None, spec=None) -> html.Div:
    options = bundle.forecast_options(sheet)
    frame, schema, _, _ = bundle.interaction_context(sheet)

    date = spec.time if spec else options["dates"][0]
    measure = spec.measure if spec else options["measures"][0]
    freq = spec.freq if spec else suggest_freq(frame[date])
    horizon = spec.horizon if spec else DEFAULT_HORIZON[freq]
    aggregation = (
        ("sum" if spec.aggregation == "count" else spec.aggregation) if spec
        else default_aggregation(schema, measure) if measure != ROW_COUNT else "sum"
    )
    group = spec.group if spec and spec.group else NO_SPLIT

    def measure_label(column: str) -> str:
        return "Number of rows" if column == ROW_COUNT else column

    return html.Div(
        [
            html.Div(
                [
                    _dropdown("fc-date", "Date", [{"label": c, "value": c} for c in options["dates"]], date),
                    _dropdown("fc-measure", "Measure", [{"label": measure_label(c), "value": c} for c in options["measures"]], measure, width="220px"),
                    _dropdown("fc-agg", "Combine values by", AGG_OPTIONS, aggregation),
                    _dropdown(
                        "fc-group", "Split by",
                        [{"label": "No split (total only)", "value": NO_SPLIT}] + [{"label": c, "value": c} for c in options["groups"]],
                        group, width="220px",
                    ),
                ],
                className="target-picker",
            ),
            html.Div(
                [
                    _dropdown("fc-freq", "Period", FREQ_OPTIONS, freq, width="140px"),
                    _number("fc-horizon", "Periods ahead", horizon, str(DEFAULT_HORIZON[freq])),
                    html.Button("Forecast", id="fc-build", n_clicks=0, className="btn btn-primary"),
                ],
                className="target-picker",
            ),
            html.P(
                "Measures recorded on each row come first; columns that are fixed "
                "per item (looked up, or constant per key) are listed last because "
                "adding them up over time means little.",
                className="card-explanation",
            ),
        ]
    )


def forecast_results_view(result) -> html.Div:
    table = result.metrics.set_index("Model")
    best = table.loc[result.best_model]
    baselines = table[table["Type"] == "baseline"]
    reference = baselines["MASE"].min() if len(baselines) else float("nan")
    total = result.forecasts[result.forecasts["Series"] == TOTAL]
    diagnostics = result.diagnostics

    tiles = [
        ui.kpi_tile("Chosen model", result.best_model, f"computed in {result.seconds:.0f}s"),
        ui.kpi_tile("Backtest MASE", f"{best['MASE']:.2f}", f"best baseline {reference:.2f} · below 1 beats seasonal naive"),
        ui.kpi_tile("Backtest sMAPE", f"{best['sMAPE']:.1%}", "average % error"),
        ui.kpi_tile(f"Next {len(total)} period(s)", format_number(total["Forecast"].sum()), result.measure_label()),
    ]
    if not np.isnan(diagnostics.get("trend_strength", np.nan)):
        tiles.append(ui.kpi_tile(
            "Trend / seasonality",
            f"{diagnostics['trend_strength']:.2f} / {diagnostics['seasonal_strength']:.2f}",
            "0 = none, 1 = strong",
        ))

    notes = [ui.message(result.summary, "success")] + [ui.message(text, "info") for text in result.notes]
    names = list(result.series)

    forecast_table = result.forecasts.copy()
    forecast_table["Period"] = forecast_table["Period"].dt.strftime("%Y-%m-%d")
    for column in ("Forecast", "Low 80%", "High 80%", "Low 95%", "High 95%"):
        forecast_table[column] = forecast_table[column].round(2)

    sections = [
        html.H2(f"Results: {result.measure_label()} by {dict((o['value'], o['label']) for o in FREQ_OPTIONS)[result.spec.freq].lower()}", className="section-title"),
        html.Div(tiles, className="kpi-row"),
        html.Div(notes),
        ui.section(
            "Forecast",
            html.Div(
                [
                    html.Div(
                        [_dropdown("fc-series", "Series", [{"label": n, "value": n} for n in names], TOTAL, width="240px")],
                        className="target-picker",
                    ),
                    html.Div(forecast_chart(result, TOTAL), id="fc-chart"),
                ],
                className="card",
            ),
        ),
        ui.section(
            "Model comparison (rolling backtest)",
            ui.data_table(result.metrics, page_size=8),
            "Each model forecast the next periods from several past cut-off "
            "points; errors are averaged over cut-offs and series. MASE below 1 "
            "means better than repeating last season's values.",
        ),
    ]

    if result.decomposition is not None:
        sections.append(ui.section("Trend and seasonality (total)", html.Div(_decomposition_card(result), className="chart-grid")))

    sections.append(
        ui.section(
            "Forecast table",
            html.Div(
                [
                    html.Div(
                        [
                            html.Button("Download forecast (CSV)", id="fc-download-button", n_clicks=0, className="btn"),
                            dcc.Download(id="fc-download"),
                        ],
                        className="target-picker",
                    ),
                    ui.data_table(forecast_table, page_size=12),
                ],
                className="card",
            ),
        )
    )

    return html.Div(sections)


def forecast_chart(result, name: str) -> html.Div:
    history = result.series.get(name)
    if history is None:
        return ui.message("Unknown series.")

    future = result.forecasts[result.forecasts["Series"] == name]
    figure = go.Figure()

    figure.add_trace(go.Scatter(
        x=list(future["Period"]) + list(future["Period"])[::-1],
        y=list(future["High 95%"]) + list(future["Low 95%"])[::-1],
        fill="toself", fillcolor="rgba(42,120,214,0.10)", line={"width": 0}, mode="lines",
        name="95% interval", hoverinfo="skip",
    ))
    figure.add_trace(go.Scatter(
        x=list(future["Period"]) + list(future["Period"])[::-1],
        y=list(future["High 80%"]) + list(future["Low 80%"])[::-1],
        fill="toself", fillcolor="rgba(42,120,214,0.22)", line={"width": 0}, mode="lines",
        name="80% interval", hoverinfo="skip",
    ))
    figure.add_trace(go.Scatter(x=history.index, y=history.values, mode="lines+markers",
                                name="Actual", line={"color": "#52514e", "width": 2}, marker={"size": 4}))
    figure.add_trace(go.Scatter(
        x=[history.index[-1]] + list(future["Period"]),
        y=[history.values[-1]] + list(future["Forecast"]),
        mode="lines+markers", name=f"Forecast ({result.best_model})",
        line={"color": SERIES_COLORS[0], "width": 3, "dash": "dot"},
    ))
    apply_theme(figure, height=380)
    figure.update_yaxes(title_text=result.measure_label())

    return ui.figure_card(
        f"{result.measure_label()} — {name}",
        figure,
        "Grey: history. Blue dotted: forecast, with shaded 80% and 95% "
        "intervals estimated from the model's backtest errors.",
    )


def _decomposition_card(result) -> html.Div:
    parts = result.decomposition
    figure = go.Figure()
    for column, color in (("Observed", "#52514e"), ("Trend", SERIES_COLORS[0]), ("Seasonal", SERIES_COLORS[1])):
        figure.add_trace(go.Scatter(x=parts.index, y=parts[column], mode="lines", name=column, line={"color": color}))
    apply_theme(figure, height=340)

    return ui.figure_card(
        "STL decomposition",
        figure,
        "The total split into a smooth trend and a repeating seasonal pattern; "
        "what is left is noise.",
    )


# ----------------------------------------------------------------------
# Segments, anomalies and correlations
# ----------------------------------------------------------------------

EACH_ROW = "__rows__"
SEGMENT_METHODS = [
    {"label": "K-means, best number of segments", "value": "auto"},
    {"label": "K-means, fixed number of segments", "value": "fixed"},
    {"label": "Low / medium / high bands of one measure", "value": "bands"},
]


def segments_panel(bundle: DatasetBundle) -> html.Div:
    sheets = bundle.segment_sheets()

    if not sheets:
        return ui.empty_state("Nothing to segment", "Segmentation needs a table with at least 20 rows and some numeric or category columns.")

    last = bundle.latest_segments
    sheet = bundle.latest_segments_sheet if last and bundle.latest_segments_sheet in sheets else sheets[0]

    picker = html.Div(
        [
            html.Label("Table", className="field-label"),
            dcc.Dropdown(
                id="seg-sheet",
                options=[{"label": name if name else bundle.name, "value": name or CURRENT_SHEET} for name in sheets],
                value=sheet or CURRENT_SHEET,
                clearable=False,
                style={"minWidth": "200px"},
            ),
        ],
        className="field",
    )

    return html.Div(
        [
            dcc.Store(id="seg-job"),
            dcc.Interval(id="seg-poll", interval=1000, disabled=True),
            ui.section(
                "Segments, unusual records and correlations",
                html.Div(
                    [
                        html.Div([picker], className="target-picker"),
                        html.Div(segments_controls_view(bundle, sheet, last.spec if last and sheet == bundle.latest_segments_sheet else None), id="seg-controls"),
                    ],
                    className="card",
                ),
                "Groups similar units into segments with readable profiles, flags "
                "the most unusual ones with reasons, and shows which features move "
                "together or repeat each other.",
            ),
            html.Div(segments_results_view(last) if last else None, id="seg-results"),
        ]
    )


def segments_controls_view(bundle: DatasetBundle, sheet: str | None, spec=None) -> html.Div:
    frame, schema, _, _ = bundle.interaction_context(sheet)
    units = bundle.segment_options(sheet, None)["units"]

    # Repeated keys (doctor x month) default to one unit per key.
    if spec is not None:
        unit = spec.unit
    else:
        unit = units[0] if units and len(frame) / frame[units[0]].nunique() >= 2 else None

    options = bundle.segment_options(sheet, unit)
    features = spec.features if spec and spec.features else options["features"]
    method = "bands" if spec and spec.method == "bands" else ("fixed" if spec and spec.k else "auto")

    unit_choices = [{"label": "Each row", "value": EACH_ROW}] + [{"label": f"Each {humanize(u)}", "value": u} for u in units]

    return html.Div(
        [
            html.Div(
                [
                    _dropdown("seg-unit", "Segment", unit_choices, unit or EACH_ROW, width="220px"),
                    _dropdown("seg-method", "Method", SEGMENT_METHODS, method, width="320px"),
                    _number("seg-k", "Segments (fixed)", spec.k if spec and spec.k else 4, "4"),
                    _dropdown(
                        "seg-band", "Measure (bands)",
                        [{"label": c, "value": c} for c in options["numeric"]],
                        (spec.band_measure if spec and spec.band_measure else (options["numeric"][0] if options["numeric"] else None)),
                        width="220px",
                    ),
                ],
                className="target-picker",
            ),
            html.Div(
                [
                    _dropdown("seg-features", "Features", [{"label": f, "value": f} for f in options["features"]], features, multi=True, width="640px"),
                ],
                className="target-picker",
            ),
            html.Div(
                [
                    _number("seg-share", "Flag as unusual (%)", round((spec.anomaly_share if spec else 0.01) * 100, 1), "1"),
                    html.Button("Find segments", id="seg-build", n_clicks=0, className="btn btn-primary"),
                ],
                className="target-picker",
            ),
            html.P(
                "Per-key units average the numbers and take the most common category "
                "of each key's rows. IDs, dates, free text and categories with many "
                "levels are not used as features.",
                className="card-explanation",
            ),
        ]
    )


def segments_results_view(result) -> html.Div:
    clustering = result.clustering
    correlation = result.correlation

    tiles = [
        ui.kpi_tile("Units", format_number(len(result.table)), f"one per {humanize(result.units.unit)}" if result.units.unit else "one per row"),
        ui.kpi_tile("Features", str(len(result.units.features)), f"{len(result.units.numeric)} numeric · {len(result.units.categorical)} category"),
        ui.kpi_tile("Segments", str(clustering.k), clustering.method),
    ]
    if clustering.silhouette is not None:
        tiles.append(ui.kpi_tile("Silhouette", f"{clustering.silhouette:.2f}", "−1 to 1, higher = better separated"))
    tiles.append(ui.kpi_tile("Unusual", str(len(result.anomalies)), f"top {result.spec.anomaly_share:.1%} by anomaly score"))

    notes = [ui.message(result.summary, "success")] + [ui.message(text, "info") for text in result.notes]

    profiles = clustering.profiles.drop(columns=["Id"]).copy()
    profiles["Share"] = (profiles["Share"] * 100).round(1).astype(str) + "%"

    charts = []
    if not clustering.projection.empty:
        charts.append(_projection_card(clustering))
    if not clustering.heatmap.empty:
        charts.append(_segment_heatmap_card(clustering))
    if clustering.k_scores is not None:
        charts.append(_silhouette_card(clustering))

    anomalies = result.anomalies.copy()
    key_columns = [c for c in anomalies.columns if c in (result.units.unit, "Row", "Segment", "Anomaly score", "Why unusual")]

    sections = [
        html.H2("Results: segments", className="section-title"),
        html.Div(tiles, className="kpi-row"),
        html.Div(notes),
        ui.section(
            "Segment profiles",
            ui.data_table(profiles, page_size=10),
            "Traits compare each segment's average with the overall average "
            "(standardised difference ≥ 0.5) and list over-represented categories.",
        ),
        ui.section("Maps and profiles", html.Div(charts, className="chart-grid")),
        ui.section(
            "Unusual units",
            html.Div(
                [
                    html.Div(
                        [
                            html.Button("Download units with segments (CSV)", id="seg-download-button", n_clicks=0, className="btn"),
                            dcc.Download(id="seg-download"),
                        ],
                        className="target-picker",
                    ),
                    ui.data_table(anomalies[key_columns], page_size=10) if len(anomalies) else ui.message("No unusual units flagged."),
                ],
                className="card",
            ),
            "Isolation Forest scores how easy each unit is to separate from the "
            "rest; reasons use robust z-scores (median and MAD) and rare categories.",
        ),
        ui.section(
            "Correlations",
            html.Div(
                ([_correlation_card(correlation)] if not correlation.matrix.empty else []),
                className="chart-grid",
            ),
            "Spearman rank correlation between numeric features (−1 to 1). "
            "Association, not causation.",
        ),
    ]

    if len(correlation.redundant):
        sections.append(ui.section("Redundant feature pairs", ui.data_table(correlation.redundant, page_size=8),
                                   "|rank correlation| ≥ 0.9: one of each pair adds little for modelling."))
    if correlation.vif is not None:
        sections.append(ui.section("Multicollinearity (VIF)", ui.data_table(correlation.vif, page_size=8),
                                   "How much of each numeric feature the others explain; above 10 is high."))
    if len(correlation.categorical):
        sections.append(ui.section("Category associations", ui.data_table(correlation.categorical, page_size=8),
                                   "Cramér's V between category features (0 = independent, 1 = one determines the other)."))

    return html.Div(sections)


def _projection_card(clustering) -> html.Div:
    figure = go.Figure()
    for index, name in enumerate(clustering.projection["Segment"].unique()):
        part = clustering.projection[clustering.projection["Segment"] == name]
        figure.add_trace(go.Scatter(
            x=part["PC1"], y=part["PC2"], mode="markers", name=name,
            marker={"size": 6, "opacity": 0.7, "color": SERIES_COLORS[index % len(SERIES_COLORS)]},
        ))
    apply_theme(figure, height=380)
    figure.update_xaxes(title_text="Component 1")
    figure.update_yaxes(title_text="Component 2")
    return ui.figure_card("Segment map", figure, "Units projected to two dimensions (PCA) and coloured by segment.")


def _segment_heatmap_card(clustering) -> html.Div:
    heat = clustering.heatmap
    figure = go.Figure(go.Heatmap(
        z=heat.to_numpy(), x=[humanize(c) for c in heat.columns], y=list(heat.index),
        colorscale=[[0, "#e34948"], [0.5, "#ffffff"], [1, "#2a78d6"]], zmid=0,
        colorbar={"title": "vs avg (σ)"},
    ))
    apply_theme(figure, height=max(260, 70 * len(heat) + 140))
    figure.update_xaxes(tickangle=-40)
    return ui.figure_card("Segment profiles", figure, "Blue = above average, red = below, in standard deviations.")


def _silhouette_card(clustering) -> html.Div:
    scores = clustering.k_scores
    figure = go.Figure(go.Scatter(x=scores["k"], y=scores["Silhouette"], mode="lines+markers", line={"color": SERIES_COLORS[0], "width": 3}))
    figure.add_vline(x=clustering.k, line_dash="dash", line_color="#9a9894")
    apply_theme(figure, height=320)
    figure.update_xaxes(title_text="Number of segments", dtick=1)
    figure.update_yaxes(title_text="Silhouette")
    return ui.figure_card("How many segments?", figure, "Silhouette for each number of segments; the dashed line marks the one used.")


def _correlation_card(correlation) -> html.Div:
    matrix = correlation.matrix
    labels = [humanize(c) for c in matrix.columns]
    figure = go.Figure(go.Heatmap(
        z=matrix.to_numpy(), x=labels, y=labels, zmin=-1, zmax=1,
        colorscale=[[0, "#e34948"], [0.5, "#ffffff"], [1, "#2a78d6"]],
        colorbar={"title": "ρ"},
    ))
    apply_theme(figure, height=max(380, 26 * len(labels) + 160))
    figure.update_xaxes(tickangle=-40)
    return ui.figure_card("Rank correlation matrix", figure, "Blue = move together, red = move in opposite directions.")


# ----------------------------------------------------------------------
# Investigate ("why did it change?")
# ----------------------------------------------------------------------

COMPARE_OPTIONS = [
    {"label": "Previous period", "value": "previous"},
    {"label": "Same period last year", "value": "year"},
]
DIRECTION_OPTIONS = [
    {"label": "Higher is better", "value": "up"},
    {"label": "Lower is better", "value": "down"},
]
NO_ENTITY = "__none__"


def investigate_panel(bundle: DatasetBundle) -> html.Div:
    sheets = bundle.forecast_sheets()

    if not sheets:
        return ui.empty_state("Nothing to investigate", "Investigation needs a table with a date column.")

    last = bundle.latest_why
    sheet = bundle.latest_why_sheet if last and bundle.latest_why_sheet in sheets else sheets[0]

    picker = html.Div(
        [
            html.Label("Table", className="field-label"),
            dcc.Dropdown(
                id="why-sheet",
                options=[{"label": name if name else bundle.name, "value": name or CURRENT_SHEET} for name in sheets],
                value=sheet or CURRENT_SHEET,
                clearable=False,
                style={"minWidth": "200px"},
            ),
        ],
        className="field",
    )

    return html.Div(
        [
            dcc.Store(id="why-job"),
            dcc.Interval(id="why-poll", interval=1000, disabled=True),
            ui.section(
                "Why did it change?",
                html.Div(
                    [
                        html.Div([picker], className="target-picker"),
                        html.Div(investigate_controls_view(bundle, sheet, last if last and sheet == bundle.latest_why_sheet else None), id="why-controls"),
                    ],
                    className="card",
                ),
                "Compares the latest complete period with the previous one (or the "
                "same period last year), finds which groups explain the change and "
                "follows the biggest one down a few levels, then flags unusual groups "
                "and ranks what needs attention. You can also ask \"why did … change?\" "
                "in the Ask tab.",
            ),
            html.Div(investigate_results_view(last) if last else None, id="why-results"),
        ]
    )


def investigate_controls_view(bundle: DatasetBundle, sheet: str | None, last=None) -> html.Div:
    options = bundle.why_options(sheet)
    frame, schema, _, _ = bundle.interaction_context(sheet)

    spec = last.spec if last else None
    date = spec.time if spec else options["dates"][0]
    measure = spec.measure if spec else options["measures"][0]
    freq = spec.freq if spec else suggest_freq(frame[date])
    aggregation = (spec.aggregation if spec and spec.aggregation != "count" else None) or (
        default_aggregation(schema, measure) if measure != ROW_COUNT else "sum"
    )
    dimensions = spec.dimensions if spec else options["dimensions"]
    entity = (last.attention_dimension if last else None) or (options["entities"][0] if options["entities"] else NO_ENTITY)
    direction = "up" if not last or last.higher_is_better else "down"

    def measure_label(column: str) -> str:
        return "Number of rows" if column == ROW_COUNT else column

    return html.Div(
        [
            html.Div(
                [
                    _dropdown("why-date", "Date", [{"label": c, "value": c} for c in options["dates"]], date),
                    _dropdown("why-measure", "Measure", [{"label": measure_label(c), "value": c} for c in options["measures"]], measure, width="220px"),
                    _dropdown("why-agg", "Combine values by", [o for o in AGG_OPTIONS if o["value"] in {"sum", "mean"}], aggregation),
                    _dropdown("why-freq", "Period", FREQ_OPTIONS, freq, width="140px"),
                    _dropdown("why-compare", "Compare with", COMPARE_OPTIONS, spec.compare if spec else "previous", width="220px"),
                ],
                className="target-picker",
            ),
            html.Div(
                [
                    _dropdown("why-dims", "Explain by", [{"label": c, "value": c} for c in options["dimensions"]], dimensions, multi=True, width="560px"),
                ],
                className="target-picker",
            ),
            html.Div(
                [
                    _dropdown(
                        "why-entity", "Rank for attention",
                        [{"label": "No ranking", "value": NO_ENTITY}] + [{"label": c, "value": c} for c in options["entities"]],
                        entity, width="220px",
                    ),
                    _dropdown("why-direction", "Direction", DIRECTION_OPTIONS, direction, width="180px"),
                    html.Button("Investigate", id="why-build", n_clicks=0, className="btn btn-primary"),
                ],
                className="target-picker",
            ),
            html.P(
                "Explain-by columns are categories and keys with up to 100 values; "
                "columns that repeat another one (a key and its name) are listed once.",
                className="card-explanation",
            ),
        ]
    )


def investigate_results_view(result) -> html.Div:
    comparison = result.comparison
    spec = result.spec
    good = (comparison.change >= 0) == result.higher_is_better

    tiles = [
        ui.kpi_tile(result.label(), format_number(comparison.current), _period_label(comparison.current_period, spec)),
        ui.kpi_tile("Compared with", format_number(comparison.previous), _period_label(comparison.previous_period, spec)),
        ui.kpi_tile("Change", f"{comparison.change:+,.2f}".rstrip("0").rstrip("."),
                    (f"{comparison.change_pct:+.1%}" if not np.isnan(comparison.change_pct) else "") + (" · good" if good else " · needs attention")),
    ]
    if len(result.attention):
        top = result.attention.iloc[0]
        tiles.append(ui.kpi_tile("Top attention", str(top[result.attention_dimension]), f"score {top['Attention score']}"))

    steps = html.Ol([html.Li(step.text, className="why-step") for step in result.steps], className="why-steps")
    notes = [ui.message(text, "info") for text in result.notes]

    charts = [_trend_card(result)]
    if result.breakdowns:
        charts.append(_waterfall_card(result, result.breakdowns[0]))

    sections = [
        html.H2(f"Results: why {result.label().lower()} changed", className="section-title"),
        html.Div(tiles, className="kpi-row"),
        ui.section("Explanation", html.Div([steps] + notes, className="card"),
                   "Each step follows the group that explains most of the movement into the next dimension."),
        ui.section("Over time and by contribution", html.Div(charts, className="chart-grid")),
    ]

    for item in result.breakdowns[:4]:
        table = item.table.head(MAX_BREAKDOWN_ROWS).copy()
        table["Share of change"] = (table["Share of change"] * 100).round(1)
        table = table.rename(columns={"Share of change": "Share of change %"})
        for column in table.columns:
            if table[column].dtype.kind == "f":
                table[column] = table[column].round(2)
        sections.append(ui.section(
            f"By {humanize(item.dimension)}",
            ui.data_table(table, page_size=10),
            f"The biggest group moving with the total accounts for {item.concentration:.0%} of all movement across {humanize(item.dimension)}.",
        ))

    if result.attention_dimension:
        sections.append(ui.section(
            f"Which {humanize(result.attention_dimension)} need attention",
            html.Div(
                [
                    html.Div(
                        [
                            html.Button("Download attention ranking (CSV)", id="why-download-button", n_clicks=0, className="btn"),
                            dcc.Download(id="why-download"),
                        ],
                        className="target-picker",
                    ),
                    ui.data_table(result.attention, page_size=10) if len(result.attention) else ui.message("Not enough history to rank."),
                ],
                className="card",
            ),
            "Score combines the change vs the comparison period, how unusual the "
            "current period is for that group, the recent trend and the group's size. "
            "'Why' lists the factors behind each score.",
        ))
        sections.append(ui.section(
            "Unusual this period",
            ui.data_table(result.unusual, page_size=10) if len(result.unusual) else ui.message("No group is far from its usual range."),
            "Groups whose current value is far from their own history (robust z ≥ 2.5).",
        ))

    return html.Div(sections)


MAX_BREAKDOWN_ROWS = 15


def _period_label(stamp, spec) -> str:
    if spec.freq == "M":
        return f"{stamp:%b %Y}"
    if spec.freq == "Q":
        return f"Q{stamp.quarter} {stamp.year}"
    if spec.freq == "W":
        return f"week of {stamp:%Y-%m-%d}"
    return f"{stamp:%Y-%m-%d}"


def _trend_card(result) -> html.Div:
    trend = result.trend
    comparison = result.comparison
    marked = [comparison.previous_period, comparison.current_period]

    figure = go.Figure()
    figure.add_trace(go.Scatter(x=trend.index, y=trend.values, mode="lines+markers", name=result.label(),
                                line={"color": "#52514e", "width": 2}, marker={"size": 5}))
    figure.add_trace(go.Scatter(x=marked, y=[trend.get(p) for p in marked], mode="markers", name="Compared periods",
                                marker={"size": 12, "color": SERIES_COLORS[0]}))
    apply_theme(figure, height=340)
    figure.update_yaxes(title_text=result.label())
    return ui.figure_card(f"{result.label()} over time", figure, "The two compared periods are highlighted.")


def _waterfall_card(result, item) -> html.Div:
    comparison = result.comparison
    table = item.table
    top = table.head(8)
    rest = table["Change"].iloc[8:].sum()

    labels = [_period_label(comparison.previous_period, result.spec)] + [str(v) for v in top[item.dimension]]
    values = [comparison.previous] + list(top["Change"])
    measures = ["absolute"] + ["relative"] * len(top)
    if len(table) > 8:
        labels.append("Other groups")
        values.append(rest)
        measures.append("relative")
    labels.append(_period_label(comparison.current_period, result.spec))
    values.append(comparison.current)
    measures.append("total")

    figure = go.Figure(go.Waterfall(
        x=labels, y=values, measure=measures,
        increasing={"marker": {"color": SERIES_COLORS[0]}},
        decreasing={"marker": {"color": "#e34948"}},
        totals={"marker": {"color": "#52514e"}},
        connector={"line": {"color": "#c9c7c1"}},
    ))
    apply_theme(figure, height=340)
    figure.update_xaxes(tickangle=-30)
    return ui.figure_card(
        f"From {labels[0]} to {labels[-1]} by {humanize(item.dimension)}",
        figure,
        "Blue raised the total, red lowered it.",
    )
