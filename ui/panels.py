"""Content builders for each dashboard tab."""

from __future__ import annotations

from typing import Any

import pandas as pd
from dash import dcc, html

from core.schema_inference import format_number, humanize
from core.target_analysis import eligible_targets
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
