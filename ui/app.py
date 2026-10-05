"""
Intelligent EDA dashboard.

    Upload any dataset (CSV / Excel / JSON / Parquet)
        -> Overview      schema, roles, data quality, preview
        -> Auto insights recommended charts + ranked findings
        -> Chart builder manual charts with auto defaults
        -> Ask           natural-language questions -> answer + chart
        -> Target        training-table EDA: target rate, leakage, split
        -> Build model   goal -> trained, compared, explained model + scores
        -> Recommend     next-best-contact model + daily plan per user + backtest
        -> Forecast      measure over time: backtested models + forecast intervals
        -> Segments      clusters with profiles, unusual records, correlations
        -> Investigate   why a number changed: drill-down, unusual groups, attention
        -> KPIs          business KPIs from an optional domain file (domains/*.yaml)
        -> My board      pinned charts from every tab
"""

from __future__ import annotations

import base64
from pathlib import Path
from urllib.parse import parse_qs
from typing import Any

import pandas as pd
from dash import ALL, MATCH, Dash, Input, Output, State, ctx, dcc, html, no_update

from core.schema_inference import format_number
from core.forecast import DEFAULT_HORIZON, ROW_COUNT, SeriesSpec, default_aggregation, suggest_freq
from core.modeling import save_model
from core.segment import SegmentSpec
from core.why import ChangeSpec
from core.recommend import PlanSettings, suggest_success
from core.target_analysis import report_markdown

from . import components as ui
from . import panels
from service.session import DatasetBundle, store


ASSETS_FOLDER = str(Path(__file__).parent / "assets")

TAB_VALUES = {"overview", "auto", "builder", "ask", "target", "model", "recommend", "forecast", "segments", "why", "kpi", "board"}

HIDDEN = {"display": "none"}
SHEET_PICKER = {"display": "flex", "alignItems": "center", "gap": "8px"}
SHOWN: dict[str, Any] = {}

# Dropdown option cap for filter values.
MAX_FILTER_OPTIONS = 300


def create_app(
    initial_file: str | None = None,
    sheet: str | None = None,
) -> Dash:

    initial_id = None

    if initial_file:
        path = Path(initial_file)
        bundle = store.load(path.read_bytes(), path.name, sheet)
        initial_id = bundle.id

    app = Dash(
        __name__,
        title="Intelligent EDA",
        assets_folder=ASSETS_FOLDER,
        suppress_callback_exceptions=True,
    )

    app.layout = _layout(initial_id)

    _register_callbacks(app)

    return app


# ----------------------------------------------------------------------
# Layout
# ----------------------------------------------------------------------

def _upload(component_id: str, children: Any, class_name: str) -> dcc.Upload:
    return dcc.Upload(
        id=component_id,
        children=children,
        multiple=False,
        accept=".csv,.tsv,.txt,.xlsx,.xls,.json,.parquet",
        className=class_name,
    )


def _layout(initial_id: str | None) -> html.Div:
    header = html.Header(
        [
            html.Div(
                [
                    html.Div("Intelligent EDA", className="brand"),
                    html.Div(id="dataset-chip", className="dataset-chip"),
                ],
                className="header-left",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("Sheet", className="field-label"),
                            dcc.Dropdown(
                                id="sheet-select",
                                clearable=False,
                                placeholder="Sheet",
                                style={"minWidth": "200px"},
                            ),
                        ],
                        id="sheet-wrap",
                        className="sheet-select",
                        style=HIDDEN,
                    ),
                    _upload(
                        "upload",
                        html.Button("Upload data", className="btn btn-primary"),
                        "upload-button",
                    ),
                ],
                className="header-right",
            ),
        ],
        className="header",
    )

    empty = html.Div(
        _upload(
            "upload-hero",
            html.Div(
                [
                    html.H1("Explore any dataset"),
                    html.P(
                        "Drop a CSV, Excel, JSON or Parquet file here, or "
                        "click to choose one. You get an automatic overview, "
                        "data-quality checks, recommended charts, a chart "
                        "builder and a question box."
                    ),
                    html.Span("Choose a file", className="btn btn-primary"),
                ],
                className="hero",
            ),
            "hero-upload",
        ),
        id="empty-wrap",
    )

    tabs = dcc.Tabs(
        id="tabs",
        value="overview",
        className="tabs",
        children=[
            dcc.Tab(
                label="Overview",
                value="overview",
                className="tab",
                selected_className="tab--selected",
                children=dcc.Loading(
                    html.Div(id="overview-panel", className="panel"),
                    type="dot",
                    color="var(--accent)",
                ),
            ),
            dcc.Tab(
                label="Auto insights",
                value="auto",
                className="tab",
                selected_className="tab--selected",
                children=dcc.Loading(
                    html.Div(id="auto-panel", className="panel"),
                    type="dot",
                    color="var(--accent)",
                ),
            ),
            dcc.Tab(
                label="Chart builder",
                value="builder",
                className="tab",
                selected_className="tab--selected",
                children=html.Div(panels.builder_panel(), className="panel"),
            ),
            dcc.Tab(
                label="Ask",
                value="ask",
                className="tab",
                selected_className="tab--selected",
                children=html.Div(id="ask-panel", className="panel"),
            ),
            dcc.Tab(
                label="Target",
                value="target",
                className="tab",
                selected_className="tab--selected",
                children=dcc.Loading(
                    html.Div(id="target-panel", className="panel"),
                    type="dot",
                    color="var(--accent)",
                ),
            ),
            dcc.Tab(
                label="Build model",
                value="model",
                className="tab",
                selected_className="tab--selected",
                children=dcc.Loading(
                    html.Div(id="model-panel", className="panel"),
                    type="dot",
                    color="var(--accent)",
                ),
            ),
            dcc.Tab(
                label="Recommend",
                value="recommend",
                className="tab",
                selected_className="tab--selected",
                children=dcc.Loading(
                    html.Div(id="recommend-panel", className="panel"),
                    type="dot",
                    color="var(--accent)",
                ),
            ),
            dcc.Tab(
                label="Forecast",
                value="forecast",
                className="tab",
                selected_className="tab--selected",
                children=dcc.Loading(
                    html.Div(id="forecast-panel", className="panel"),
                    type="dot",
                    color="var(--accent)",
                ),
            ),
            dcc.Tab(
                label="Segments",
                value="segments",
                className="tab",
                selected_className="tab--selected",
                children=dcc.Loading(
                    html.Div(id="segments-panel", className="panel"),
                    type="dot",
                    color="var(--accent)",
                ),
            ),
            dcc.Tab(
                label="Investigate",
                value="why",
                className="tab",
                selected_className="tab--selected",
                children=dcc.Loading(
                    html.Div(id="why-panel", className="panel"),
                    type="dot",
                    color="var(--accent)",
                ),
            ),
            dcc.Tab(
                label="KPIs",
                value="kpi",
                className="tab",
                selected_className="tab--selected",
                children=dcc.Loading(
                    html.Div(id="kpi-panel", className="panel"),
                    type="dot",
                    color="var(--accent)",
                ),
            ),
            dcc.Tab(
                id="board-tab",
                label="My board",
                value="board",
                className="tab",
                selected_className="tab--selected",
                children=html.Div(id="board-panel", className="panel"),
            ),
        ],
    )

    return html.Div(
        [
            dcc.Location(id="url", refresh=False),
            dcc.Store(id="dataset-id", data=initial_id),
            dcc.Store(id="board", data=[]),
            dcc.Store(id="auto-owner", data=None),
            dcc.Store(id="target-owner", data=None),
            dcc.Store(id="model-owner", data=None),
            dcc.Store(id="recommend-owner", data=None),
            dcc.Store(id="forecast-owner", data=None),
            dcc.Store(id="segments-owner", data=None),
            dcc.Store(id="why-owner", data=None),
            dcc.Store(id="kpi-owner", data=None),
            header,
            html.Main(
                [
                    html.Div(id="upload-message"),
                    empty,
                    html.Div(tabs, id="tabs-wrap", style=HIDDEN),
                ],
                className="main",
            ),
        ],
        className="app",
    )


# ----------------------------------------------------------------------
# Callbacks
# ----------------------------------------------------------------------

def _register_callbacks(app: Dash) -> None:

    # -- Loading data ----------------------------------------------------

    @app.callback(
        Output("dataset-id", "data"),
        Output("sheet-select", "options"),
        Output("sheet-select", "value"),
        Output("sheet-wrap", "style"),
        Output("upload-message", "children"),
        Output("upload", "contents"),
        Output("upload-hero", "contents"),
        Input("upload", "contents"),
        Input("upload-hero", "contents"),
        Input("sheet-select", "value"),
        State("upload", "filename"),
        State("upload-hero", "filename"),
        State("dataset-id", "data"),
    )
    def load_data(contents, hero_contents, sheet, filename, hero_filename, dataset_id):
        trigger = ctx.triggered_id
        bundle = store.get(dataset_id)

        try:
            if trigger in {"upload", "upload-hero"}:
                raw = contents if trigger == "upload" else hero_contents
                name = filename if trigger == "upload" else hero_filename

                if not raw:
                    return (no_update,) * 7

                bundle = store.load(_decode(raw), name)

            elif trigger == "sheet-select" and bundle and sheet != bundle.sheet:
                bundle = store.load(bundle.content, bundle.name, sheet)

        except Exception as error:  # noqa: BLE001 - shown to the user
            return (
                no_update, no_update, no_update, no_update,
                ui.message(f"Could not load the file: {error}", "error"),
                None, None,
            )

        if bundle is None:
            return None, [], None, HIDDEN, None, None, None

        options = [{"label": name, "value": name} for name in bundle.sheets]
        style = SHEET_PICKER if len(bundle.sheets) > 1 else HIDDEN

        return bundle.id, options, bundle.sheet, style, None, None, None

    # -- Deep links: ?tab=auto|builder|ask|board -------------------------

    @app.callback(
        Output("tabs", "value"),
        Input("url", "search"),
    )
    def open_tab_from_url(search):
        tab = parse_qs((search or "").lstrip("?")).get("tab", [None])[0]
        return tab if tab in TAB_VALUES else no_update

    # -- Dataset-wide content --------------------------------------------

    @app.callback(
        Output("dataset-chip", "children"),
        Output("empty-wrap", "style"),
        Output("tabs-wrap", "style"),
        Output("overview-panel", "children"),
        Output("ask-panel", "children"),
        Input("dataset-id", "data"),
    )
    def show_dataset(dataset_id):
        bundle = store.get(dataset_id)

        if bundle is None:
            return None, SHOWN, HIDDEN, None, None

        chip = [
            html.Span(bundle.name, className="chip-name"),
            html.Span(
                f"{format_number(len(bundle.df))} rows × "
                f"{bundle.df.shape[1]} columns"
                + (
                    f" · sheet {bundle.sheet} "
                    f"({bundle.sheets.index(bundle.sheet) + 1} of {len(bundle.sheets)})"
                    if bundle.sheet in bundle.sheets
                    else ""
                ),
                className="chip-meta",
            ),
        ]

        overview = _safe(panels.overview_panel, bundle)
        ask = _safe(panels.ask_panel, bundle)

        return chip, HIDDEN, SHOWN, overview, ask

    @app.callback(
        Output("auto-panel", "children"),
        Output("auto-owner", "data"),
        Input("tabs", "value"),
        Input("dataset-id", "data"),
        State("auto-owner", "data"),
    )
    def show_auto(tab, dataset_id, owner):
        # Built lazily the first time the tab is opened for a dataset.
        if tab != "auto" or dataset_id is None or owner == dataset_id:
            return no_update, no_update

        bundle = store.get(dataset_id)

        if bundle is None:
            return None, None

        return _safe(panels.auto_panel, bundle), dataset_id

    # -- Target -----------------------------------------------------------

    @app.callback(
        Output("target-panel", "children"),
        Output("target-owner", "data"),
        Input("tabs", "value"),
        Input("dataset-id", "data"),
        State("target-owner", "data"),
    )
    def show_target(tab, dataset_id, owner):
        # Built lazily the first time the tab is opened for a dataset.
        if tab != "target" or dataset_id is None or owner == dataset_id:
            return no_update, no_update

        bundle = store.get(dataset_id)

        if bundle is None:
            return None, None

        return _safe(panels.target_panel, bundle), dataset_id

    @app.callback(
        Output("target-body", "children"),
        Input("target-select", "value"),
        State("dataset-id", "data"),
        prevent_initial_call=True,
    )
    def change_target(target, dataset_id):
        bundle = store.get(dataset_id)

        if bundle is None:
            return no_update

        return _safe(panels.target_body, bundle, target)

    @app.callback(
        Output("target-download", "data"),
        Input("target-download-button", "n_clicks"),
        State("target-select", "value"),
        State("dataset-id", "data"),
        prevent_initial_call=True,
    )
    def download_target_report(clicks, target, dataset_id):
        bundle = store.get(dataset_id)

        if not clicks or bundle is None or not target:
            return no_update

        try:
            report = bundle.target_report(target)
        except ValueError:
            return no_update

        name = bundle.sheet or Path(bundle.name).stem
        text = report_markdown(report, f"{bundle.name} ({name})" if bundle.sheet else bundle.name)

        return dcc.send_string(text, f"target_report_{target}.md")

    # -- Build model ------------------------------------------------------

    @app.callback(
        Output("model-panel", "children"),
        Output("model-owner", "data"),
        Input("tabs", "value"),
        Input("dataset-id", "data"),
        State("model-owner", "data"),
    )
    def show_model(tab, dataset_id, owner):
        if tab != "model" or dataset_id is None or owner == dataset_id:
            return no_update, no_update

        bundle = store.get(dataset_id)

        if bundle is None:
            return None, None

        return _safe(panels.model_panel, bundle), dataset_id

    @app.callback(
        Output("model-target", "options"),
        Output("model-target", "value"),
        Input("model-goal", "value"),
        State("dataset-id", "data"),
        prevent_initial_call=True,
    )
    def change_model_goal(goal, dataset_id):
        bundle = store.get(dataset_id)

        if bundle is None or not goal:
            return no_update, no_update

        return panels.model_target_options(bundle, goal)

    @app.callback(
        Output("model-setup", "children"),
        Input("model-target", "value"),
        State("model-goal", "value"),
        State("dataset-id", "data"),
        prevent_initial_call=True,
    )
    def change_model_target(target, goal, dataset_id):
        bundle = store.get(dataset_id)

        if bundle is None:
            return no_update

        return _safe(panels.model_setup_view, bundle, goal, target)

    @app.callback(
        Output("model-job", "data"),
        Output("model-poll", "disabled"),
        Output("model-results", "children"),
        Input("model-train", "n_clicks"),
        State("model-goal", "value"),
        State("model-target", "value"),
        State("dataset-id", "data"),
        prevent_initial_call=True,
    )
    def train_model(clicks, goal, target, dataset_id):
        bundle = store.get(dataset_id)

        if not clicks or bundle is None or not goal or not target:
            return no_update, no_update, no_update

        try:
            job_id = bundle.start_model_job(goal, target)
        except ValueError as error:
            return None, True, ui.message(str(error), "error")

        return job_id, False, panels.model_progress_view(bundle.model_job(job_id))

    @app.callback(
        Output("model-results", "children", allow_duplicate=True),
        Output("model-poll", "disabled", allow_duplicate=True),
        Input("model-poll", "n_intervals"),
        State("model-job", "data"),
        State("dataset-id", "data"),
        prevent_initial_call=True,
    )
    def poll_model(_ticks, job_id, dataset_id):
        bundle = store.get(dataset_id)
        job = bundle.model_job(job_id) if bundle else None

        if job is None:
            return no_update, True

        if job["status"] == "running":
            return panels.model_progress_view(job), False

        if job["status"] == "error":
            return ui.message(f"Training failed: {job['error']}", "error"), True

        return _safe(panels.model_results_view, job["result"]), True

    @app.callback(
        Output("model-download", "data"),
        Input("model-download-button", "n_clicks"),
        State("dataset-id", "data"),
        prevent_initial_call=True,
    )
    def download_predictions(clicks, dataset_id):
        bundle = store.get(dataset_id)

        if not clicks or bundle is None or bundle.latest_model is None:
            return no_update

        result = bundle.latest_model
        return dcc.send_data_frame(
            result.scored.to_csv,
            f"predictions_{result.spec.target}.csv",
            index=False,
        )

    @app.callback(
        Output("model-save-message", "children"),
        Output("model-saved", "children"),
        Input("model-save-button", "n_clicks"),
        State("dataset-id", "data"),
        prevent_initial_call=True,
    )
    def save_trained_model(clicks, dataset_id):
        bundle = store.get(dataset_id)

        if not clicks or bundle is None or bundle.latest_model is None:
            return no_update, no_update

        try:
            name = bundle.name if not bundle.sheet else f"{bundle.name} ({bundle.sheet})"
            meta = save_model(bundle.latest_model, bundle.df, name)
        except Exception as error:  # noqa: BLE001 - shown to the user
            return f"Could not save: {error}", no_update

        return f"Saved as {meta['name']}", panels.saved_models_view()

    # -- Recommend ----------------------------------------------------------

    @app.callback(
        Output("recommend-panel", "children"),
        Output("recommend-owner", "data"),
        Input("tabs", "value"),
        Input("dataset-id", "data"),
        State("recommend-owner", "data"),
    )
    def show_recommend(tab, dataset_id, owner):
        if tab != "recommend" or dataset_id is None or owner == dataset_id:
            return no_update, no_update

        bundle = store.get(dataset_id)

        if bundle is None:
            return None, None

        return _safe(panels.recommend_panel, bundle), dataset_id

    @app.callback(
        Output("rec-roles", "children"),
        Input("rec-sheet", "value"),
        State("dataset-id", "data"),
        prevent_initial_call=True,
    )
    def change_rec_sheet(sheet, dataset_id):
        bundle = store.get(dataset_id)

        if bundle is None:
            return no_update

        return _safe(panels.recommend_roles_view, bundle, panels.sheet_from_value(sheet))

    @app.callback(
        Output("rec-success", "options"),
        Output("rec-success", "value"),
        Input("rec-outcome", "value"),
        State("rec-sheet", "value"),
        State("dataset-id", "data"),
        prevent_initial_call=True,
    )
    def change_rec_outcome(outcome, sheet, dataset_id):
        bundle = store.get(dataset_id)

        if bundle is None or not outcome:
            return no_update, no_update

        frame, _, _, _ = bundle.interaction_context(panels.sheet_from_value(sheet))
        return panels.success_options(frame, outcome), suggest_success(frame[outcome])

    @app.callback(
        Output("rec-group", "options"),
        Output("rec-group", "value"),
        Input("rec-user", "value"),
        Input("rec-item", "value"),
        State("rec-sheet", "value"),
        State("dataset-id", "data"),
        prevent_initial_call=True,
    )
    def change_rec_sides(user, item, sheet, dataset_id):
        bundle = store.get(dataset_id)

        if bundle is None or not user or not item:
            return no_update, no_update

        groups = bundle.interaction_groups(panels.sheet_from_value(sheet), user, item)
        options = panels.group_dropdown(groups)
        return options, options[0]["value"]

    @app.callback(
        Output("rec-job", "data"),
        Output("rec-poll", "disabled"),
        Output("rec-results", "children"),
        Input("rec-build", "n_clicks"),
        State("rec-sheet", "value"),
        State("rec-user", "value"),
        State("rec-item", "value"),
        State("rec-time", "value"),
        State("rec-outcome", "value"),
        State("rec-success", "value"),
        State("rec-group", "value"),
        State("rec-period", "value"),
        State("rec-days", "value"),
        State("rec-start", "date"),
        State("rec-capacity", "value"),
        State("rec-gap", "value"),
        State("rec-owner", "value"),
        State("dataset-id", "data"),
        prevent_initial_call=True,
    )
    def build_plan(clicks, sheet, user, item, time_column, outcome, success, group,
                   period, days, start, capacity, gap, owner, dataset_id):
        bundle = store.get(dataset_id)

        if not clicks or bundle is None:
            return no_update, no_update, no_update

        sheet = panels.sheet_from_value(sheet)

        try:
            roles = bundle.interaction_roles(
                sheet,
                user=user,
                item=item,
                time=time_column,
                outcome=outcome,
                success_values=success or [],
                group=None if group == panels.NO_GROUP_VALUE else group,
            )
            if period == "month":
                plan_days = int(days or 22)
            elif period == "custom":
                plan_days = int(days or 5)
            else:
                plan_days = int(period or 5)
            settings = PlanSettings(
                start=pd.Timestamp(start) if start else None,
                days=plan_days,
                capacity=int(capacity) if capacity else None,
                min_gap_days=int(gap) if gap else None,
                period="month" if period == "month" else "days",
                one_owner="owner" in (owner or []),
            )
            job_id = bundle.start_recommend_job(sheet, roles, settings)
        except (ValueError, TypeError, KeyError) as error:
            return None, True, ui.message(str(error), "error")

        return job_id, False, panels.recommend_progress_view(bundle.model_job(job_id))

    @app.callback(
        Output("rec-results", "children", allow_duplicate=True),
        Output("rec-poll", "disabled", allow_duplicate=True),
        Input("rec-poll", "n_intervals"),
        State("rec-job", "data"),
        State("dataset-id", "data"),
        prevent_initial_call=True,
    )
    def poll_plan(_ticks, job_id, dataset_id):
        bundle = store.get(dataset_id)
        job = bundle.model_job(job_id) if bundle else None

        if job is None:
            return no_update, True

        if job["status"] == "running":
            return panels.recommend_progress_view(job), False

        if job["status"] == "error":
            return ui.message(f"Planning failed: {job['error']}", "error"), True

        return _safe(panels.recommend_results_view, job["result"]), True

    @app.callback(
        Output("rec-user-plan", "children"),
        Input("rec-user-view", "value"),
        State("dataset-id", "data"),
        prevent_initial_call=True,
    )
    def change_rec_user(user, dataset_id):
        bundle = store.get(dataset_id)

        if bundle is None or bundle.latest_recommendation is None or not user:
            return no_update

        return _safe(panels.recommend_user_plan, bundle.latest_recommendation, user)

    @app.callback(
        Output("rec-download", "data"),
        Input("rec-download-button", "n_clicks"),
        State("dataset-id", "data"),
        prevent_initial_call=True,
    )
    def download_plan(clicks, dataset_id):
        bundle = store.get(dataset_id)

        if not clicks or bundle is None or bundle.latest_recommendation is None:
            return no_update

        return dcc.send_data_frame(bundle.latest_recommendation.plan.to_csv, "visit_plan.csv", index=False)

    # -- Forecast -----------------------------------------------------------

    @app.callback(
        Output("forecast-panel", "children"),
        Output("forecast-owner", "data"),
        Input("tabs", "value"),
        Input("dataset-id", "data"),
        State("forecast-owner", "data"),
    )
    def show_forecast(tab, dataset_id, owner):
        if tab != "forecast" or dataset_id is None or owner == dataset_id:
            return no_update, no_update

        bundle = store.get(dataset_id)

        if bundle is None:
            return None, None

        return _safe(panels.forecast_panel, bundle), dataset_id

    @app.callback(
        Output("fc-controls", "children"),
        Input("fc-sheet", "value"),
        State("dataset-id", "data"),
        prevent_initial_call=True,
    )
    def change_fc_sheet(sheet, dataset_id):
        bundle = store.get(dataset_id)

        if bundle is None:
            return no_update

        return _safe(panels.forecast_controls_view, bundle, panels.sheet_from_value(sheet))

    @app.callback(
        Output("fc-freq", "value"),
        Output("fc-horizon", "value"),
        Input("fc-date", "value"),
        State("fc-sheet", "value"),
        State("dataset-id", "data"),
        prevent_initial_call=True,
    )
    def change_fc_date(date, sheet, dataset_id):
        bundle = store.get(dataset_id)

        if bundle is None or not date:
            return no_update, no_update

        frame, _, _, _ = bundle.interaction_context(panels.sheet_from_value(sheet))
        freq = suggest_freq(frame[date])
        return freq, DEFAULT_HORIZON[freq]

    @app.callback(
        Output("fc-agg", "value"),
        Input("fc-measure", "value"),
        State("fc-sheet", "value"),
        State("dataset-id", "data"),
        prevent_initial_call=True,
    )
    def change_fc_measure(measure, sheet, dataset_id):
        bundle = store.get(dataset_id)

        if bundle is None or not measure or measure == ROW_COUNT:
            return "sum"

        _, schema, _, _ = bundle.interaction_context(panels.sheet_from_value(sheet))
        return default_aggregation(schema, measure)

    @app.callback(
        Output("fc-job", "data"),
        Output("fc-poll", "disabled"),
        Output("fc-results", "children"),
        Input("fc-build", "n_clicks"),
        State("fc-sheet", "value"),
        State("fc-date", "value"),
        State("fc-measure", "value"),
        State("fc-agg", "value"),
        State("fc-group", "value"),
        State("fc-freq", "value"),
        State("fc-horizon", "value"),
        State("dataset-id", "data"),
        prevent_initial_call=True,
    )
    def build_forecast_job(clicks, sheet, date, measure, aggregation, group, freq, horizon, dataset_id):
        bundle = store.get(dataset_id)

        if not clicks or bundle is None or not date or not measure:
            return no_update, no_update, no_update

        spec = SeriesSpec(
            time=date,
            measure=measure,
            aggregation="count" if measure == ROW_COUNT else (aggregation or "sum"),
            group=None if group in (None, panels.NO_SPLIT) else group,
            freq=freq or "M",
            horizon=int(horizon or DEFAULT_HORIZON.get(freq or "M", 3)),
        )

        try:
            job_id = bundle.start_forecast_job(panels.sheet_from_value(sheet), spec)
        except (ValueError, TypeError, KeyError) as error:
            return None, True, ui.message(str(error), "error")

        return job_id, False, panels.model_progress_view(bundle.model_job(job_id))

    @app.callback(
        Output("fc-results", "children", allow_duplicate=True),
        Output("fc-poll", "disabled", allow_duplicate=True),
        Input("fc-poll", "n_intervals"),
        State("fc-job", "data"),
        State("dataset-id", "data"),
        prevent_initial_call=True,
    )
    def poll_forecast(_ticks, job_id, dataset_id):
        bundle = store.get(dataset_id)
        job = bundle.model_job(job_id) if bundle else None

        if job is None:
            return no_update, True

        if job["status"] == "running":
            return panels.model_progress_view(job), False

        if job["status"] == "error":
            return ui.message(f"Forecast failed: {job['error']}", "error"), True

        return _safe(panels.forecast_results_view, job["result"]), True

    @app.callback(
        Output("fc-chart", "children"),
        Input("fc-series", "value"),
        State("dataset-id", "data"),
        prevent_initial_call=True,
    )
    def change_fc_series(name, dataset_id):
        bundle = store.get(dataset_id)

        if bundle is None or bundle.latest_forecast is None or not name:
            return no_update

        return _safe(panels.forecast_chart, bundle.latest_forecast, name)

    @app.callback(
        Output("fc-download", "data"),
        Input("fc-download-button", "n_clicks"),
        State("dataset-id", "data"),
        prevent_initial_call=True,
    )
    def download_forecast(clicks, dataset_id):
        bundle = store.get(dataset_id)

        if not clicks or bundle is None or bundle.latest_forecast is None:
            return no_update

        return dcc.send_data_frame(bundle.latest_forecast.forecasts.to_csv, "forecast.csv", index=False)

    # -- Segments -----------------------------------------------------------

    @app.callback(
        Output("segments-panel", "children"),
        Output("segments-owner", "data"),
        Input("tabs", "value"),
        Input("dataset-id", "data"),
        State("segments-owner", "data"),
    )
    def show_segments(tab, dataset_id, owner):
        if tab != "segments" or dataset_id is None or owner == dataset_id:
            return no_update, no_update

        bundle = store.get(dataset_id)

        if bundle is None:
            return None, None

        return _safe(panels.segments_panel, bundle), dataset_id

    @app.callback(
        Output("seg-controls", "children"),
        Input("seg-sheet", "value"),
        State("dataset-id", "data"),
        prevent_initial_call=True,
    )
    def change_seg_sheet(sheet, dataset_id):
        bundle = store.get(dataset_id)

        if bundle is None:
            return no_update

        return _safe(panels.segments_controls_view, bundle, panels.sheet_from_value(sheet))

    @app.callback(
        Output("seg-features", "options"),
        Output("seg-features", "value"),
        Output("seg-band", "options"),
        Output("seg-band", "value"),
        Input("seg-unit", "value"),
        State("seg-sheet", "value"),
        State("dataset-id", "data"),
        prevent_initial_call=True,
    )
    def change_seg_unit(unit, sheet, dataset_id):
        bundle = store.get(dataset_id)

        if bundle is None:
            return no_update, no_update, no_update, no_update

        options = bundle.segment_options(panels.sheet_from_value(sheet), None if unit == panels.EACH_ROW else unit)
        features = [{"label": f, "value": f} for f in options["features"]]
        numeric = [{"label": f, "value": f} for f in options["numeric"]]
        return features, options["features"], numeric, (options["numeric"][0] if options["numeric"] else None)

    @app.callback(
        Output("seg-job", "data"),
        Output("seg-poll", "disabled"),
        Output("seg-results", "children"),
        Input("seg-build", "n_clicks"),
        State("seg-sheet", "value"),
        State("seg-unit", "value"),
        State("seg-method", "value"),
        State("seg-k", "value"),
        State("seg-band", "value"),
        State("seg-features", "value"),
        State("seg-share", "value"),
        State("dataset-id", "data"),
        prevent_initial_call=True,
    )
    def build_segments_job(clicks, sheet, unit, method, k, band, features, share, dataset_id):
        bundle = store.get(dataset_id)

        if not clicks or bundle is None:
            return no_update, no_update, no_update

        spec = SegmentSpec(
            unit=None if unit in (None, panels.EACH_ROW) else unit,
            features=features or None,
            method="bands" if method == "bands" else "kmeans",
            k=int(k) if method == "fixed" and k else None,
            band_measure=band,
            anomaly_share=min(max(float(share or 1) / 100, 0.001), 0.2),
        )

        try:
            job_id = bundle.start_segment_job(panels.sheet_from_value(sheet), spec)
        except (ValueError, TypeError, KeyError) as error:
            return None, True, ui.message(str(error), "error")

        return job_id, False, panels.model_progress_view(bundle.model_job(job_id))

    @app.callback(
        Output("seg-results", "children", allow_duplicate=True),
        Output("seg-poll", "disabled", allow_duplicate=True),
        Input("seg-poll", "n_intervals"),
        State("seg-job", "data"),
        State("dataset-id", "data"),
        prevent_initial_call=True,
    )
    def poll_segments(_ticks, job_id, dataset_id):
        bundle = store.get(dataset_id)
        job = bundle.model_job(job_id) if bundle else None

        if job is None:
            return no_update, True

        if job["status"] == "running":
            return panels.model_progress_view(job), False

        if job["status"] == "error":
            return ui.message(f"Segmentation failed: {job['error']}", "error"), True

        return _safe(panels.segments_results_view, job["result"]), True

    @app.callback(
        Output("seg-download", "data"),
        Input("seg-download-button", "n_clicks"),
        State("dataset-id", "data"),
        prevent_initial_call=True,
    )
    def download_segments(clicks, dataset_id):
        bundle = store.get(dataset_id)

        if not clicks or bundle is None or bundle.latest_segments is None:
            return no_update

        return dcc.send_data_frame(bundle.latest_segments.table.to_csv, "segments.csv", index=False)

    # -- Investigate --------------------------------------------------------

    @app.callback(
        Output("why-panel", "children"),
        Output("why-owner", "data"),
        Input("tabs", "value"),
        Input("dataset-id", "data"),
        State("why-owner", "data"),
    )
    def show_why(tab, dataset_id, owner):
        if tab != "why" or dataset_id is None or owner == dataset_id:
            return no_update, no_update

        bundle = store.get(dataset_id)

        if bundle is None:
            return None, None

        return _safe(panels.investigate_panel, bundle), dataset_id

    @app.callback(
        Output("why-controls", "children"),
        Input("why-sheet", "value"),
        State("dataset-id", "data"),
        prevent_initial_call=True,
    )
    def change_why_sheet(sheet, dataset_id):
        bundle = store.get(dataset_id)

        if bundle is None:
            return no_update

        return _safe(panels.investigate_controls_view, bundle, panels.sheet_from_value(sheet))

    @app.callback(
        Output("why-agg", "value"),
        Input("why-measure", "value"),
        State("why-sheet", "value"),
        State("dataset-id", "data"),
        prevent_initial_call=True,
    )
    def change_why_measure(measure, sheet, dataset_id):
        bundle = store.get(dataset_id)

        if bundle is None or not measure or measure == ROW_COUNT:
            return "sum"

        _, schema, _, _ = bundle.interaction_context(panels.sheet_from_value(sheet))
        return default_aggregation(schema, measure)

    @app.callback(
        Output("why-job", "data"),
        Output("why-poll", "disabled"),
        Output("why-results", "children"),
        Input("why-build", "n_clicks"),
        State("why-sheet", "value"),
        State("why-date", "value"),
        State("why-measure", "value"),
        State("why-agg", "value"),
        State("why-freq", "value"),
        State("why-compare", "value"),
        State("why-dims", "value"),
        State("why-entity", "value"),
        State("why-direction", "value"),
        State("dataset-id", "data"),
        prevent_initial_call=True,
    )
    def build_why_job(clicks, sheet, date, measure, aggregation, freq, compare, dims, entity, direction, dataset_id):
        bundle = store.get(dataset_id)

        if not clicks or bundle is None or not date or not measure:
            return no_update, no_update, no_update

        spec = ChangeSpec(
            time=date,
            measure=measure,
            aggregation="count" if measure == ROW_COUNT else (aggregation or "sum"),
            freq=freq or "M",
            compare=compare or "previous",
            dimensions=dims or [],
        )

        try:
            job_id = bundle.start_why_job(
                panels.sheet_from_value(sheet), spec,
                None if entity in (None, panels.NO_ENTITY) else entity,
                direction != "down",
            )
        except (ValueError, TypeError, KeyError) as error:
            return None, True, ui.message(str(error), "error")

        return job_id, False, panels.model_progress_view(bundle.model_job(job_id))

    @app.callback(
        Output("why-results", "children", allow_duplicate=True),
        Output("why-poll", "disabled", allow_duplicate=True),
        Input("why-poll", "n_intervals"),
        State("why-job", "data"),
        State("dataset-id", "data"),
        prevent_initial_call=True,
    )
    def poll_why(_ticks, job_id, dataset_id):
        bundle = store.get(dataset_id)
        job = bundle.model_job(job_id) if bundle else None

        if job is None:
            return no_update, True

        if job["status"] == "running":
            return panels.model_progress_view(job), False

        if job["status"] == "error":
            return ui.message(f"Investigation failed: {job['error']}", "error"), True

        return _safe(panels.investigate_results_view, job["result"]), True

    @app.callback(
        Output("why-download", "data"),
        Input("why-download-button", "n_clicks"),
        State("dataset-id", "data"),
        prevent_initial_call=True,
    )
    def download_attention(clicks, dataset_id):
        bundle = store.get(dataset_id)

        if not clicks or bundle is None or bundle.latest_why is None:
            return no_update

        return dcc.send_data_frame(bundle.latest_why.attention.to_csv, "attention_ranking.csv", index=False)

    # -- KPIs ---------------------------------------------------------------

    @app.callback(
        Output("kpi-panel", "children"),
        Output("kpi-owner", "data"),
        Input("tabs", "value"),
        Input("dataset-id", "data"),
        State("kpi-owner", "data"),
    )
    def show_kpis(tab, dataset_id, owner):
        if tab != "kpi" or dataset_id is None or owner == dataset_id:
            return no_update, no_update

        bundle = store.get(dataset_id)

        if bundle is None:
            return None, None

        return _safe(panels.kpi_panel, bundle), dataset_id

    @app.callback(
        Output("kpi-group", "options"),
        Output("kpi-group", "value"),
        Input("kpi-domain", "value"),
        State("dataset-id", "data"),
    )
    def change_kpi_domain(path, dataset_id):
        bundle = store.get(dataset_id)

        if bundle is None or not path:
            return no_update, no_update

        try:
            return panels.kpi_group_options(bundle, path), panels.NO_GROUP_ROLE
        except (ValueError, OSError):
            return no_update, no_update

    @app.callback(
        Output("kpi-results", "children"),
        Input("kpi-domain", "value"),
        Input("kpi-freq", "value"),
        Input("kpi-group", "value"),
        State("dataset-id", "data"),
        prevent_initial_call=True,
    )
    def change_kpi_view(path, freq, group, dataset_id):
        bundle = store.get(dataset_id)

        if bundle is None or not path:
            return no_update

        role = None if group in (None, panels.NO_GROUP_ROLE) else group
        return _safe(panels.kpi_results_view, bundle, path, freq or "M", role)

    @app.callback(
        Output("kpi-trend", "children"),
        Input("kpi-trend-select", "value"),
        State("kpi-domain", "value"),
        State("kpi-freq", "value"),
        State("dataset-id", "data"),
        prevent_initial_call=True,
    )
    def change_kpi_trend(label, path, freq, dataset_id):
        bundle = store.get(dataset_id)

        if bundle is None or not path or not label:
            return no_update

        report, _ = bundle.kpi_report(path, freq or "M", None)
        return _safe(panels.kpi_trend_card, report, label, report.summary)

    @app.callback(
        Output("kpi-download", "data"),
        Input("kpi-download-button", "n_clicks"),
        State("kpi-domain", "value"),
        State("kpi-freq", "value"),
        State("kpi-group", "value"),
        State("dataset-id", "data"),
        prevent_initial_call=True,
    )
    def download_kpis(clicks, path, freq, group, dataset_id):
        bundle = store.get(dataset_id)

        if not clicks or bundle is None or not path:
            return no_update

        role = None if group in (None, panels.NO_GROUP_ROLE) else group
        report, _ = bundle.kpi_report(path, freq or "M", role)
        table = report.by_group if report.by_group is not None else report.summary.drop(columns=["_id"])
        return dcc.send_data_frame(table.to_csv, "kpis.csv", index=False)

    @app.callback(
        Output("kpi-starter", "data"),
        Input("kpi-starter-button", "n_clicks"),
        State("dataset-id", "data"),
        prevent_initial_call=True,
    )
    def download_starter(clicks, dataset_id):
        bundle = store.get(dataset_id)

        if not clicks or bundle is None:
            return no_update

        return dcc.send_string(bundle.kpi_starter(), "starter_domain.yaml")

    # -- Chart builder ----------------------------------------------------

    @app.callback(
        Output("b-x", "options"),
        Output("b-y", "options"),
        Output("b-color", "options"),
        Output("b-filter-col", "options"),
        Output("b-y", "disabled"),
        Output("b-x", "value"),
        Output("b-y", "value"),
        Output("b-color", "value"),
        Output("b-filter-col", "value"),
        Input("b-type", "value"),
        Input("dataset-id", "data"),
        State("b-x", "value"),
        State("b-y", "value"),
        State("b-color", "value"),
    )
    def builder_options(chart_type, dataset_id, x_value, y_value, color_value):
        bundle = store.get(dataset_id)

        if bundle is None:
            return [], [], [], [], False, None, None, None, None

        x_options, y_options = _builder_axis_options(bundle, chart_type)

        color_options = panels.column_options(
            bundle, {"dimension", "binary"}, include_high_cardinality=False,
        )
        filter_options = panels.column_options(
            bundle, {"dimension", "binary", "measure", "identifier"},
        )

        y_disabled = chart_type == "HISTOGRAM"

        if ctx.triggered_id == "dataset-id" or ctx.triggered_id is None:
            x_value, y_value = _default_builder_columns(bundle)
            color_value = None
            filter_value = None
        else:
            filter_value = no_update

        x_value = _keep(x_value, x_options)
        y_value = None if y_disabled else _keep(y_value, y_options)
        color_value = _keep(color_value, color_options)

        return (
            x_options, y_options, color_options, filter_options, y_disabled,
            x_value, y_value, color_value, filter_value,
        )

    @app.callback(
        Output("b-filter-values", "options"),
        Output("b-filter-values", "value"),
        Output("b-filter-values-wrap", "style"),
        Output("b-filter-range-wrap", "style"),
        Output("b-filter-min", "value"),
        Output("b-filter-max", "value"),
        Output("b-filter-min", "placeholder"),
        Output("b-filter-max", "placeholder"),
        Input("b-filter-col", "value"),
        State("dataset-id", "data"),
    )
    def filter_widgets(column, dataset_id):
        bundle = store.get(dataset_id)

        if bundle is None or not column or column not in bundle.df.columns:
            return [], [], SHOWN, HIDDEN, None, None, None, None

        series = bundle.df[column]

        if bundle.schema.role_of(column) == "measure":
            return (
                [], [], HIDDEN, {"display": "flex", "gap": "12px"}, None, None,
                f"min {format_number(series.min())}",
                f"max {format_number(series.max())}",
            )

        values = series.dropna().value_counts().index[:MAX_FILTER_OPTIONS]
        options = [{"label": str(value), "value": _json_safe(value)} for value in values]

        return options, [], SHOWN, HIDDEN, None, None, None, None

    @app.callback(
        Output("b-output", "children"),
        Input("b-type", "value"),
        Input("b-x", "value"),
        Input("b-y", "value"),
        Input("b-agg", "value"),
        Input("b-color", "value"),
        Input("b-grain", "value"),
        Input("b-limit", "value"),
        Input("b-filter-col", "value"),
        Input("b-filter-values", "value"),
        Input("b-filter-min", "value"),
        Input("b-filter-max", "value"),
        State("dataset-id", "data"),
    )
    def build_chart(
        chart_type, x_column, y_column, aggregation, color_column,
        time_grain, limit, filter_column, filter_values, filter_min,
        filter_max, dataset_id,
    ):
        bundle = store.get(dataset_id)

        if bundle is None:
            return None

        if not x_column and not y_column:
            return ui.message("Choose at least one column to draw a chart.")

        filters = _builder_filters(
            bundle, filter_column, filter_values, filter_min, filter_max,
        )

        try:
            spec = bundle.build_manual(
                chart_type=chart_type,
                x_column=x_column,
                y_column=y_column,
                aggregation=aggregation,
                color_column=color_column,
                limit=limit,
                filters=filters,
                time_grain=time_grain,
            )
            rendered = bundle.render(spec)
        except (ValueError, TypeError, KeyError) as error:
            return ui.message(str(error), "error")

        return ui.chart_card(rendered, explanation=spec.reason)

    # -- Ask --------------------------------------------------------------

    @app.callback(
        Output("ask-output", "children"),
        Output("ask-input", "value"),
        Input("ask-button", "n_clicks"),
        Input("ask-input", "n_submit"),
        Input({"type": "suggest", "index": ALL}, "n_clicks"),
        State("ask-input", "value"),
        State("dataset-id", "data"),
        prevent_initial_call=True,
    )
    def ask(_clicks, _submits, _suggest_clicks, question, dataset_id):
        bundle = store.get(dataset_id)

        if bundle is None:
            return no_update, no_update

        trigger = ctx.triggered_id

        if isinstance(trigger, dict) and trigger.get("type") == "suggest":
            if not ctx.triggered[0]["value"]:
                return no_update, no_update
            question = panels.suggest_questions(bundle.df, bundle.schema)[trigger["index"]]

        if not question or not question.strip():
            return no_update, no_update

        bundle.ask(question.strip())

        return panels.answer_cards(bundle), question

    # -- Board --------------------------------------------------------------

    @app.callback(
        Output("board", "data"),
        Input({"type": "pin", "key": ALL, "uid": ALL}, "n_clicks"),
        Input({"type": "unpin", "key": ALL, "uid": ALL}, "n_clicks"),
        Input("board-clear", "n_clicks"),
        Input("dataset-id", "data"),
        State("board", "data"),
        prevent_initial_call=True,
    )
    def update_board(_pins, _unpins, _clear, _dataset_id, board):
        trigger = ctx.triggered_id
        board = list(board or [])

        if trigger == "dataset-id":
            return []

        if trigger == "board-clear":
            return [] if ctx.triggered[0]["value"] else no_update

        # New buttons appearing also fire this callback with 0 clicks.
        if not isinstance(trigger, dict) or not ctx.triggered[0]["value"]:
            return no_update

        key = trigger["key"]

        if trigger["type"] == "pin" and key not in board:
            board.append(key)
        elif trigger["type"] == "unpin" and key in board:
            board.remove(key)
        else:
            return no_update

        return board

    @app.callback(
        Output({"type": "pin", "key": MATCH, "uid": MATCH}, "children"),
        Input({"type": "pin", "key": MATCH, "uid": MATCH}, "n_clicks"),
        prevent_initial_call=True,
    )
    def pin_feedback(n_clicks):
        return "Pinned ✓" if n_clicks else no_update

    @app.callback(
        Output("board-panel", "children"),
        Output("board-tab", "label"),
        Input("board", "data"),
        Input("dataset-id", "data"),
    )
    def show_board(board, dataset_id):
        bundle = store.get(dataset_id)
        keys = board or []
        label = f"My board ({len(keys)})" if keys else "My board"

        return _safe(panels.board_panel, bundle, keys), label


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _decode(contents: str) -> bytes:
    _, encoded = contents.split(",", 1)
    return base64.b64decode(encoded)


def _safe(builder, *args) -> Any:
    """Render a panel; show the error inline instead of breaking the page."""

    try:
        return builder(*args)
    except Exception as error:  # noqa: BLE001 - shown to the user
        return ui.message(f"Something went wrong building this view: {error}", "error")


def _keep(value: Any, options: list[dict[str, Any]]) -> Any:
    values = {option["value"] for option in options}
    return value if value in values else None


def _json_safe(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _builder_axis_options(bundle: DatasetBundle, chart_type: str):
    numeric = {"measure"}
    numeric_or_rate = {"measure", "binary"}
    categorical = {"dimension", "binary", "time"}
    everything = {"measure", "dimension", "binary", "time", "identifier"}

    if chart_type == "HISTOGRAM":
        return panels.column_options(bundle, numeric), []

    if chart_type == "SCATTER":
        return (
            panels.column_options(bundle, numeric),
            panels.column_options(bundle, numeric),
        )

    if chart_type == "BOX":
        return (
            panels.column_options(bundle, {"dimension", "binary"}),
            panels.column_options(bundle, numeric),
        )

    if chart_type in {"BAR", "LINE", "PIE"}:
        return (
            panels.column_options(bundle, categorical | {"identifier", "measure"}),
            panels.column_options(bundle, numeric_or_rate | {"identifier", "dimension"}),
        )

    return (
        panels.column_options(bundle, everything),
        panels.column_options(bundle, everything),
    )


def _default_builder_columns(bundle: DatasetBundle) -> tuple[str | None, str | None]:
    schema = bundle.schema
    x_column = (schema.dimensions or schema.binaries or [schema.primary_time] or [None])[0]
    measures = schema.aggregatable
    y_column = next((m for m in measures if m != x_column), None)
    return x_column, y_column


def _builder_filters(bundle, column, values, minimum, maximum) -> list[dict[str, Any]]:
    if not column or column not in bundle.df.columns:
        return []

    if bundle.schema.role_of(column) == "measure":
        filters = []
        if minimum is not None:
            filters.append({"column": column, "operator": ">=", "value": minimum})
        if maximum is not None:
            filters.append({"column": column, "operator": "<=", "value": maximum})
        return filters

    if values:
        return [{"column": column, "operator": "in", "value": list(values)}]

    return []
