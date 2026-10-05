"""Reusable Dash building blocks."""

from __future__ import annotations

import uuid
from typing import Any

import pandas as pd
from dash import dash_table, dcc, html

from core.schema_inference import format_number, humanize


GRAPH_CONFIG = {
    "displaylogo": False,
    "responsive": True,
    "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
    "toImageButtonOptions": {"format": "png", "scale": 2},
}

SEVERITY_LABELS = {
    "critical": ("▲", "Critical"),
    "warning": ("●", "Warning"),
    "info": ("○", "Info"),
}


def section(title: str, children: Any, subtitle: str | None = None) -> html.Div:
    header = [html.H2(title, className="section-title")]

    if subtitle:
        header.append(html.P(subtitle, className="section-subtitle"))

    return html.Div(
        [html.Div(header, className="section-header"), children],
        className="section",
    )


def kpi_tile(label: str, value: str, note: str | None = None) -> html.Div:
    children = [
        html.Div(label, className="kpi-label"),
        html.Div(value, className="kpi-value"),
    ]

    if note:
        children.append(html.Div(note, className="kpi-note"))

    return html.Div(children, className="kpi")


def message(text: str, kind: str = "info") -> html.Div:
    return html.Div(text, className=f"message message-{kind}")


def empty_state(title: str, body: str) -> html.Div:
    return html.Div(
        [html.H3(title), html.P(body)],
        className="empty-state",
    )


def chart_card(
    rendered,
    explanation: str | None = None,
    action: str = "pin",
    badge: str | None = None,
) -> html.Div:
    """
    A chart with title, explanation, footnotes, a pin/remove action and
    a collapsible data table (so values are readable without the plot).
    """

    spec = rendered.spec

    header = [
        html.Div(
            [
                html.H3(spec.title, className="card-title"),
                html.Span(badge, className="badge") if badge else None,
            ],
            className="card-title-row",
        ),
        _action_button(rendered.key, action),
    ]

    children = [html.Div(header, className="card-header")]

    if explanation:
        children.append(html.P(explanation, className="card-explanation"))

    children.append(
        dcc.Graph(
            figure=rendered.figure,
            config=GRAPH_CONFIG,
            className="card-graph",
        )
    )

    notes = _chart_notes(rendered.metadata)

    if notes:
        children.append(html.P(notes, className="card-note"))

    children.append(_data_details(rendered.data))

    return html.Div(children, className="card chart-card")


def chart_error_card(title: str, error: str) -> html.Div:
    return html.Div(
        [
            html.H3(title, className="card-title"),
            message(f"This chart could not be created: {error}", "error"),
        ],
        className="card chart-card",
    )


def data_table(
    dataframe: pd.DataFrame,
    page_size: int = 10,
    table_id: str | None = None,
) -> dash_table.DataTable:

    frame = dataframe.copy()

    for column in frame.columns:
        if pd.api.types.is_datetime64_any_dtype(frame[column]):
            frame[column] = frame[column].dt.strftime("%Y-%m-%d")

    frame.columns = [str(column) for column in frame.columns]

    kwargs = {"id": table_id} if table_id else {}

    return dash_table.DataTable(
        data=frame.to_dict("records"),
        columns=[
            {
                "name": column,
                "id": column,
                "type": "numeric" if pd.api.types.is_numeric_dtype(frame[column]) else "text",
            }
            for column in frame.columns
        ],
        page_size=page_size,
        sort_action="native",
        style_as_list_view=True,
        style_table={"overflowX": "auto"},
        style_header={
            "backgroundColor": "var(--surface-2)",
            "fontWeight": "600",
            "color": "var(--text-primary)",
            "border": "none",
            "borderBottom": "1px solid var(--border)",
        },
        style_cell={
            "fontFamily": "var(--font)",
            "fontSize": "13px",
            "padding": "8px 12px",
            "color": "var(--text-primary)",
            "backgroundColor": "var(--surface-1)",
            "border": "none",
            "borderBottom": "1px solid var(--border)",
            "maxWidth": "320px",
            "overflow": "hidden",
            "textOverflow": "ellipsis",
            "textAlign": "left",
        },
        style_cell_conditional=[
            {"if": {"column_type": "numeric"}, "textAlign": "right"},
        ],
        tooltip_delay=0,
        **kwargs,
    )


def issue_row(issue: dict[str, Any]) -> html.Div:
    icon, label = SEVERITY_LABELS[issue["severity"]]

    return html.Div(
        [
            html.Span(
                [html.Span(icon, className="severity-icon"), label],
                className=f"severity severity-{issue['severity']}",
            ),
            html.Span(issue["message"], className="issue-text"),
        ],
        className="issue",
    )


def pin_id(key: str) -> dict[str, str]:
    """
    Pattern-matching id for a pin button. The same chart can appear in
    several places, so each button gets a unique suffix.
    """
    return {"type": "pin", "key": key, "uid": _uid()}


# ----------------------------------------------------------------------
# Internals
# ----------------------------------------------------------------------

def _uid() -> str:
    return uuid.uuid4().hex[:8]


def _action_button(key: str, action: str) -> html.Button | None:
    if action == "pin":
        return html.Button(
            "Pin to board",
            id=pin_id(key),
            n_clicks=0,
            className="btn btn-ghost",
            title="Add this chart to My Board",
        )

    if action == "unpin":
        return html.Button(
            "Remove",
            id={"type": "unpin", "key": key, "uid": _uid()},
            n_clicks=0,
            className="btn btn-ghost",
        )

    return None


def _chart_notes(metadata: dict[str, Any]) -> str:
    notes = []

    if metadata.get("truncated_from"):
        notes.append(
            f"Showing {metadata.get('category_count', '')} of "
            f"{format_number(metadata['truncated_from'])} categories."
        )

    if metadata.get("sampled_from"):
        notes.append(
            f"Random sample of {format_number(metadata['row_count'])} of "
            f"{format_number(metadata['sampled_from'])} points."
        )

    if metadata.get("time_grain"):
        grain = {
            "D": "day", "W": "week", "M": "month", "Q": "quarter", "Y": "year",
        }.get(metadata["time_grain"], metadata["time_grain"])
        notes.append(f"Grouped by {grain}.")

    source_rows = metadata.get("source_rows")

    if source_rows is not None:
        notes.append(f"Based on {format_number(source_rows)} rows.")

    return " ".join(notes)


def _data_details(data: list[dict[str, Any]]) -> html.Details:
    frame = pd.DataFrame(data)

    if len(frame) > 500:
        frame = frame.head(500)

    frame.columns = [humanize(column) for column in frame.columns]

    return html.Details(
        [
            html.Summary("View data", className="details-summary"),
            data_table(frame, page_size=8),
        ],
        className="card-details",
    )
