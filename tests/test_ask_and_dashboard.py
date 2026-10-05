import json

import plotly

from core.schema_inference import infer_schema
from tests.sample_data import retail_dataframe
from ui.app import create_app
from ui.state import DatasetBundle
from ui.suggestions import suggest_questions


def _bundle():
    df = retail_dataframe()
    return DatasetBundle(
        id="test",
        name="retail.csv",
        content=b"",
        sheets=[],
        sheet=None,
        df=df,
        schema=infer_schema(df),
    )


def test_every_suggested_question_is_answerable():
    bundle = _bundle()
    questions = suggest_questions(bundle.df, bundle.schema)

    assert len(questions) >= 4

    for question in questions:
        entry = bundle.ask(question)
        assert entry["error"] is None, (question, entry["error"])
        assert entry["answer"]


def test_ranking_question_produces_matching_chart():
    bundle = _bundle()
    entry = bundle.ask("Which are the top 2 regions by Sales?")

    assert entry["error"] is None
    spec = bundle.charts[entry["chart_key"]]
    assert (spec.chart_type, spec.x_column, spec.limit) == ("BAR", "Region", 2)

    rendered = bundle.render(entry["chart_key"])
    assert len({row["Region"] for row in rendered.data}) == 2


def test_which_subject_resolves_from_dataset_columns():
    bundle = _bundle()
    entry = bundle.ask("Which region has the highest Sales?")

    assert entry["error"] is None
    assert "Region" in bundle.charts[entry["chart_key"]].x_column


def test_count_answer_names_the_entity_generically():
    bundle = _bundle()
    entry = bundle.ask("How many customers are there?")

    assert entry["error"] is None
    assert "customers" in entry["answer"]


def test_insights_and_panels_render():
    from ui import panels

    bundle = _bundle()

    for builder in (panels.overview_panel, panels.auto_panel, panels.ask_panel):
        component = builder(bundle)
        json.dumps(component, cls=plotly.utils.PlotlyJSONEncoder)

    assert bundle.insights()
    assert all("chart_key" in insight for insight in bundle.insights())


def test_app_layout_builds(tmp_path):
    path = tmp_path / "retail.csv"
    retail_dataframe(300).to_csv(path, index=False)

    app = create_app(str(path))

    assert app.layout is not None
    assert len(app.callback_map) >= 8
