import pandas as pd
import pytest

from core.schema_inference import infer_schema
from tests.sample_data import retail_dataframe
from visualization import (
    ChartEngine,
    ChartRecommender,
    ChartRenderer,
    ChartSpec,
    ManualChartBuilder,
)


@pytest.fixture(scope="module")
def data():
    df = retail_dataframe()
    return df, infer_schema(df)


def test_recommendations_are_diverse_and_renderable(data):
    df, schema = data
    recommendations = ChartRecommender().recommend_default_charts(df, 6, schema)

    assert 3 <= len(recommendations) <= 6

    categories = [rec.category for rec in recommendations]
    assert max(categories.count(c) for c in set(categories)) <= 2

    signatures = {
        (r.chart_spec.chart_type, r.chart_spec.x_column, r.chart_spec.y_column)
        for r in recommendations
    }
    assert len(signatures) == len(recommendations)

    engine, renderer = ChartEngine(), ChartRenderer()

    for rec in recommendations:
        assert rec.explanation
        assert "Total Total" not in rec.chart_spec.title
        renderer.render(engine.create_chart(df, rec.chart_spec))

    # Identifiers and free text never become chart axes.
    used = {r.chart_spec.x_column for r in recommendations}
    assert not used & {"OrderID", "customer_id", "Phone", "Comment"}


def test_recommender_finds_the_planted_group_difference(data):
    df, schema = data
    recommendations = ChartRecommender().recommend_default_charts(df, 6, schema)

    assert any(
        r.chart_spec.x_column == "Category"
        and r.chart_spec.y_column in {"UnitPrice", "Sales"}
        for r in recommendations
    )


def test_engine_color_split_sort_and_limit(data):
    df, _ = data
    spec = ChartSpec(
        chart_type="BAR",
        x_column="Region",
        y_column="Sales",
        aggregation="SUM",
        color_column="Category",
        limit=2,
    )

    result = ChartEngine().create_chart(df, spec)
    frame = pd.DataFrame(result.data)

    assert frame["Region"].nunique() == 2
    assert set(frame["Category"]) == {"Furniture", "Office Supplies", "Technology"}

    totals = df.groupby("Region")["Sales"].sum().sort_values(ascending=False)
    assert list(dict.fromkeys(frame["Region"])) == list(totals.index[:2])


def test_engine_time_bucketing(data):
    df, _ = data
    spec = ChartSpec(
        chart_type="LINE",
        x_column="order_date",
        y_column="Sales",
        aggregation="SUM",
        time_grain="AUTO",
    )

    result = ChartEngine().create_chart(df, spec)

    assert result.metadata["time_grain"] == "W"
    assert result.metadata["row_count"] <= 110


def test_engine_in_filter(data):
    df, _ = data
    spec = ChartSpec(
        chart_type="BAR",
        x_column="Region",
        aggregation="COUNT",
        filters=[{"column": "Region", "operator": "in", "value": ["North", "South"]}],
    )

    frame = pd.DataFrame(ChartEngine().create_chart(df, spec).data)

    assert set(frame["Region"]) == {"North", "South"}
    assert frame["Count"].sum() == df["Region"].isin(["North", "South"]).sum()


def test_validator_messages(data):
    df, _ = data
    engine = ChartEngine()

    with pytest.raises(ValueError, match="not numeric"):
        engine.create_chart(df, ChartSpec("BAR", "Region", "Category", "SUM"))

    with pytest.raises(ValueError, match="numeric X"):
        engine.create_chart(df, ChartSpec("SCATTER", "Region", "Sales"))


def test_manual_builder_auto_modes(data):
    df, schema = data
    builder = ManualChartBuilder()

    spec = builder.build(df, "AUTO", "Region", "Sales", schema=schema)
    assert spec.chart_type == "BAR"
    assert spec.aggregation == "AVERAGE"

    spec = builder.build(df, "AUTO", "Quantity", "Sales", schema=schema)
    assert spec.chart_type == "SCATTER"

    spec = builder.build(df, "AUTO", "order_date", "Quantity", schema=schema)
    assert spec.chart_type == "LINE"
    assert spec.aggregation == "SUM"

    spec = builder.build(df, "BAR", "Region", "customer_id", schema=schema)
    assert spec.aggregation == "COUNT_DISTINCT"

    spec = builder.build(df, "HISTOGRAM", None, "Sales", schema=schema)
    assert spec.x_column == "Sales"
