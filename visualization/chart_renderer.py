from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from core.schema_inference import humanize

from .chart_schema import ChartResult, value_label


# Validated categorical palette (fixed order, never cycled).
SERIES_COLORS = [
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
]

PRIMARY_COLOR = SERIES_COLORS[0]

TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID_COLOR = "#e8e7e3"
SURFACE = "#ffffff"

FONT_FAMILY = (
    "Inter, -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"
)


def apply_theme(figure: go.Figure, height: int = 380) -> go.Figure:
    """The dashboard look for any Plotly figure."""

    figure.update_layout(
        title=None,
        template="plotly_white",
        height=height,
        margin={"l": 56, "r": 16, "t": 16, "b": 48},
        font={"family": FONT_FAMILY, "size": 12, "color": TEXT_SECONDARY},
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        hoverlabel={
            "bgcolor": SURFACE,
            "bordercolor": GRID_COLOR,
            "font": {"color": TEXT_PRIMARY, "family": FONT_FAMILY},
        },
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0,
            "title": {"text": ""},
        },
        barcornerradius=4,
        bargap=0.25,
    )

    figure.update_xaxes(
        showgrid=False,
        linecolor=GRID_COLOR,
        tickfont={"color": TEXT_SECONDARY},
        title_font={"color": TEXT_SECONDARY},
        automargin=True,
    )

    figure.update_yaxes(
        gridcolor=GRID_COLOR,
        zeroline=False,
        tickfont={"color": TEXT_SECONDARY},
        title_font={"color": TEXT_SECONDARY},
        automargin=True,
    )

    return figure


class ChartRenderer:
    """
    Renders ChartResult objects into Plotly figures.

    Responsibilities:
        ChartRecommender -> decides what chart to create
        ChartEngine      -> prepares chart-ready data
        ChartRenderer    -> creates the visual chart
    """

    def __init__(self, height: int = 380):
        self.height = height

    def render(self, chart_result: ChartResult) -> go.Figure:

        if chart_result is None:
            raise ValueError("ChartResult cannot be None.")

        if not chart_result.data:
            raise ValueError("ChartResult contains no data to render.")

        dataframe = pd.DataFrame(chart_result.data)

        renderers = {
            "BAR": self._render_bar,
            "LINE": self._render_line,
            "SCATTER": self._render_scatter,
            "HISTOGRAM": self._render_histogram,
            "BOX": self._render_box,
            "PIE": self._render_pie,
        }

        chart_type = chart_result.chart_spec.chart_type

        if chart_type not in renderers:
            raise ValueError(f"Unsupported chart type: '{chart_type}'")

        figure = renderers[chart_type](dataframe, chart_result)

        return self._finalize_figure(figure, chart_result)

    # ------------------------------------------------------------------
    # Chart types
    # ------------------------------------------------------------------

    def _render_bar(self, dataframe, result):
        spec = result.chart_spec

        figure = px.bar(
            dataframe,
            x=spec.x_column,
            y=result.y_column,
            color=spec.color_column,
            barmode="group",
            labels=self._labels(result),
            **self._color_args(spec.color_column),
        )

        # Keep the engine's ordering of categories.
        if not self._is_continuous(dataframe[spec.x_column]):
            figure.update_xaxes(
                type="category",
                categoryorder="array",
                categoryarray=list(dict.fromkeys(dataframe[spec.x_column])),
            )

        return figure

    def _render_line(self, dataframe, result):
        spec = result.chart_spec

        point_count = dataframe[spec.x_column].nunique()

        figure = px.line(
            dataframe,
            x=spec.x_column,
            y=result.y_column,
            color=spec.color_column,
            markers=point_count <= 40,
            labels=self._labels(result),
            **self._color_args(spec.color_column),
        )

        figure.update_traces(line={"width": 2}, marker={"size": 8})

        return figure

    def _render_scatter(self, dataframe, result):
        spec = result.chart_spec

        opacity = 0.75 if len(dataframe) < 1000 else 0.4

        figure = px.scatter(
            dataframe,
            x=spec.x_column,
            y=spec.y_column,
            color=spec.color_column,
            opacity=opacity,
            labels=self._labels(result),
            **self._color_args(spec.color_column),
        )

        figure.update_traces(marker={"size": 8, "line": {"width": 0}})

        return figure

    def _render_histogram(self, dataframe, result):
        spec = result.chart_spec

        figure = px.histogram(
            dataframe,
            x=spec.x_column,
            color=spec.color_column,
            barmode="overlay" if spec.color_column else "relative",
            opacity=0.7 if spec.color_column else 1.0,
            labels=self._labels(result),
            **self._color_args(spec.color_column),
        )

        figure.update_layout(bargap=0.04, yaxis_title="Count")

        return figure

    def _render_box(self, dataframe, result):
        spec = result.chart_spec

        if spec.x_column:
            figure = px.box(
                dataframe,
                x=spec.x_column,
                y=spec.y_column,
                labels=self._labels(result),
                color_discrete_sequence=[PRIMARY_COLOR],
            )
            figure.update_xaxes(type="category")
        else:
            figure = px.box(
                dataframe,
                y=spec.y_column,
                labels=self._labels(result),
                color_discrete_sequence=[PRIMARY_COLOR],
            )

        return figure

    def _render_pie(self, dataframe, result):
        spec = result.chart_spec

        figure = px.pie(
            dataframe,
            names=spec.x_column,
            values=result.y_column,
            hole=0.55,
            color_discrete_sequence=SERIES_COLORS,
        )

        figure.update_traces(
            textposition="outside",
            textinfo="label+percent",
            marker={"line": {"color": SURFACE, "width": 2}},
            sort=False,
        )

        return figure

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _color_args(color_column):
        if color_column:
            return {"color_discrete_sequence": SERIES_COLORS}
        return {"color_discrete_sequence": [PRIMARY_COLOR]}

    @staticmethod
    def _is_continuous(series: pd.Series) -> bool:
        return (
            pd.api.types.is_datetime64_any_dtype(series)
            or (
                pd.api.types.is_numeric_dtype(series)
                and series.nunique() > 12
            )
        )

    @staticmethod
    def _labels(result: ChartResult) -> dict[str, str]:
        spec = result.chart_spec

        labels = {}

        for column in (spec.x_column, spec.color_column):
            if column:
                labels[column] = humanize(column)

        if spec.chart_type in {"BAR", "LINE", "PIE"} and result.y_column:
            labels[result.y_column] = value_label(
                spec.aggregation, spec.y_column,
            )
        elif spec.y_column:
            labels[spec.y_column] = humanize(spec.y_column)

        return labels

    def _finalize_figure(
        self,
        figure: go.Figure,
        chart_result: ChartResult,
    ) -> go.Figure:

        apply_theme(figure, self.height)

        if chart_result.chart_spec.chart_type == "HISTOGRAM":
            figure.update_layout(bargap=0.04)

        return figure
