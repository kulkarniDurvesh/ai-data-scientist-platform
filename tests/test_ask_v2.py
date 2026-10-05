"""
Phase 1.5 Ask box: lists, date filters, chart requests, fuzzy matching.

Uses synthetic datasets with unrelated column names, so nothing here
depends on one project's schema.
"""

import io

import pandas as pd

from core.NLP.query_parser import QueryParser
from core.schema_inference import infer_schema
from tests.sample_data import retail_dataframe
from ui.state import DatasetBundle, DatasetStore


def _bundle(df=None):
    df = retail_dataframe() if df is None else df
    return DatasetBundle(
        id="test",
        name="data.csv",
        content=b"",
        sheets=[],
        sheet=None,
        df=df,
        schema=infer_schema(df),
    )


def test_month_filter_lists_distinct_entities_and_flags_years():
    bundle = _bundle()
    entry = bundle.ask("list customers who ordered in March")

    assert entry["error"] is None
    df = bundle.df
    expected = df.loc[df["order_date"].dt.month == 3, "customer_id"].nunique()

    table = entry["table"]
    assert list(table.columns)[0] == "customer_id"
    assert len(table) == expected == table["customer_id"].nunique()
    assert "customer_name" in table.columns
    # The data spans two years and no year was given.
    assert "2023" in entry["answer"] and "2024" in entry["answer"]


def test_month_and_year_filter():
    bundle = _bundle()
    entry = bundle.ask("list customers who ordered in March 2024")

    df = bundle.df
    dates = df["order_date"]
    expected = df.loc[(dates.dt.month == 3) & (dates.dt.year == 2024), "customer_id"].nunique()

    assert entry["error"] is None
    assert len(entry["table"]) == expected
    assert "March 2024" in entry["answer"]
    assert "Add a year" not in entry["answer"]


def test_date_column_is_chosen_from_question_words():
    df = retail_dataframe(500)
    df["ship_date"] = df["order_date"] + pd.Timedelta(days=40)
    parser = QueryParser(df, schema=infer_schema(df))

    shipped = parser.parse("list customers shipped in May 2024")
    ordered = parser.parse("list customers who ordered in May 2024")

    assert {f.column for f in shipped.filters if f.operator == "month"} == {"ship_date"}
    assert {f.column for f in ordered.filters if f.operator == "month"} == {"order_date"}


def test_which_question_is_a_list():
    bundle = _bundle()
    entry = bundle.ask("Which customers are in North region")

    assert entry["error"] is None
    df = bundle.df
    expected = df.loc[df["Region"] == "North", "customer_id"].nunique()
    assert len(entry["table"]) == expected
    assert entry["answer"].startswith(f"Found {expected} customers")


def test_pie_chart_request_with_misspelt_column():
    bundle = _bundle()
    entry = bundle.ask("In North, give me a pie chart of customers according to their categry")

    assert entry["error"] is None, entry["error"]
    spec = bundle.charts[entry["chart_key"]]
    assert (spec.chart_type, spec.x_column, spec.y_column, spec.aggregation) == (
        "PIE", "Category", "customer_id", "COUNT_DISTINCT",
    )
    assert spec.filters == [{"column": "Region", "operator": "=", "value": "North"}]

    north = bundle.df[bundle.df["Region"] == "North"]
    expected = north.groupby("Category")["customer_id"].nunique().to_dict()
    rendered = bundle.render(entry["chart_key"])
    assert {row["Category"]: row[spec.y_column] for row in rendered.data} == expected


def test_line_chart_by_calendar_unit_uses_the_date_column():
    bundle = _bundle()
    entry = bundle.ask("line chart of Sales by month")

    assert entry["error"] is None, entry["error"]
    spec = bundle.charts[entry["chart_key"]]
    assert (spec.chart_type, spec.x_column, spec.time_grain) == ("LINE", "order_date", "M")
    assert len(bundle.render(entry["chart_key"]).data) == 24


def test_id_codes_match_without_leading_zeros():
    bundle = _bundle()
    code = bundle.df["customer_id"].iloc[0]          # e.g. CUST-0123
    short = f"cust-{int(code.split('-')[1])}"        # e.g. cust-123

    entry = bundle.ask(f"list orders for {short}")

    assert entry["error"] is None and entry["notice"] is None
    expected = (bundle.df["customer_id"] == code).sum()
    assert len(entry["table"]) == expected


def test_unknown_code_gives_a_notice_not_an_error():
    bundle = _bundle()
    entry = bundle.ask("list customers handled by EMP07")

    assert entry["error"] is None
    assert entry["answer"] is None
    assert "EMP07" in entry["notice"]


def test_notice_names_the_sheet_that_has_the_value():
    accounts = pd.DataFrame({
        "AccountId": [f"AC{i:03d}" for i in range(40)],
        "Segment": ["Retail", "Corporate"] * 20,
    })
    reps = pd.DataFrame({
        "RepCode": [f"RP{i:03d}" for i in range(5)],
        "Zone": ["A", "B", "C", "D", "E"],
    })
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer) as writer:
        accounts.to_excel(writer, sheet_name="Accounts", index=False)
        reps.to_excel(writer, sheet_name="Reps", index=False)

    bundle = DatasetStore().load(buffer.getvalue(), "book.xlsx", "Accounts")
    entry = bundle.ask("list accounts handled by RP01")

    assert entry["error"] is None
    assert "'Reps'" in entry["notice"] and "RepCode" in entry["notice"]


def test_month_without_a_date_column_is_a_friendly_error():
    df = retail_dataframe(200).drop(columns=["order_date"])
    entry = _bundle(df).ask("how many customers ordered in March")

    assert "no date column" in entry["error"]


def test_large_numbers_stay_numeric_filters_not_years():
    df = retail_dataframe(300)
    parsed = QueryParser(df, schema=infer_schema(df)).parse(
        "show customers with Sales more than 2000"
    )

    assert [(f.column, f.operator, f.value) for f in parsed.filters] == [("Sales", ">", 2000)]


def test_names_ending_in_id_are_not_keys_unless_identifiers():
    df = retail_dataframe(400).rename(columns={"Sales": "AmountPaid"})
    bundle = _bundle(df)

    assert bundle.schema.role_of("AmountPaid") == "measure"

    chart = bundle.ask("bar chart of AmountPaid by Region")
    assert chart["error"] is None, chart["error"]
    assert bundle.charts[chart["chart_key"]].aggregation == "SUM"

    listed = bundle.ask("list customers with AmountPaid more than 500")
    assert listed["error"] is None
    assert list(listed["table"].columns)[0] == "customer_id"

    count = bundle.ask("How many customers are there?")
    assert "customers" in count["answer"]


def test_glued_words_are_split_unless_they_are_columns():
    bundle = _bundle()
    entry = bundle.ask("List all customersId from NorthRegion")

    assert entry["error"] is None, entry["error"]
    df = bundle.df
    expected = df.loc[df["Region"] == "North", "customer_id"].nunique()
    assert len(entry["table"]) == expected

    # A column typed in CamelCase is kept as one word.
    kept = bundle.ask("What is the average UnitPrice?")
    assert kept["error"] is None and "Unit Price" in kept["answer"]


def test_list_question_matching_nothing_is_not_every_row():
    entry = _bundle().ask("list all the gizmos")

    assert entry["table"] is None
    assert "couldn't match" in entry["error"]
