import io

import pandas as pd

from core.data_quality import run_quality_checks
from core.loader import load_dataset_from_bytes
from core.schema_inference import humanize, infer_schema, name_tokens
from tests.sample_data import retail_dataframe


def test_roles_are_inferred_without_column_specific_rules():
    schema = infer_schema(retail_dataframe())
    roles = {name: column.role for name, column in schema.columns.items()}

    assert roles["OrderID"] == "identifier"
    assert roles["customer_id"] == "identifier"
    assert roles["Phone"] == "identifier"
    assert roles["order_date"] == "time"
    assert roles["Region"] == "dimension"
    assert roles["Category"] == "dimension"
    assert roles["Returned"] == "binary"
    assert roles["Sales"] == "measure"
    assert roles["Quantity"] == "measure"
    assert roles["Comment"] == "constant"
    assert schema.primary_time == "order_date"


def test_repeated_entity_ids_are_identifiers_not_dimensions():
    # Panel data: each id repeats, so uniqueness alone cannot detect it.
    df = pd.DataFrame({
        "StoreId": [f"S{i % 50}" for i in range(1000)],
        "Revenue": range(1000),
    })

    schema = infer_schema(df)

    assert schema.columns["StoreId"].role == "identifier"


def test_high_cardinality_labels_are_not_used_for_auto_grouping():
    schema = infer_schema(retail_dataframe())

    assert schema.columns["customer_name"].high_cardinality
    assert "customer_name" not in schema.dimensions


def test_numeric_binary_is_a_rate():
    df = pd.DataFrame({"Converted": [0, 1] * 50, "Spend": range(100)})
    schema = infer_schema(df)

    assert schema.numeric_binaries == ["Converted"]
    assert "Converted" in schema.aggregatable


def test_loader_converts_text_dates_and_numbers():
    csv = (
        "when,amount,label,clock\n"
        "June 2025,\"1,200\",a,12:30\n"
        "July 2025,300,b,09:15\n"
        "August 2025,450,c,17:45\n"
    ).encode()

    df = load_dataset_from_bytes(csv, "data.csv")

    assert pd.api.types.is_datetime64_any_dtype(df["when"])
    assert pd.api.types.is_numeric_dtype(df["amount"])
    assert df["amount"].iloc[0] == 1200
    assert not pd.api.types.is_datetime64_any_dtype(df["clock"])


def test_loader_picks_largest_excel_sheet():
    buffer = io.BytesIO()

    with pd.ExcelWriter(buffer) as writer:
        pd.DataFrame({"Item": ["a"], "Note": ["readme"]}).to_excel(
            writer, sheet_name="README", index=False,
        )
        retail_dataframe(100).to_excel(writer, sheet_name="Data", index=False)

    df = load_dataset_from_bytes(buffer.getvalue(), "book.xlsx")

    assert len(df) == 100


def test_quality_flags_impossible_negatives_and_imbalance():
    df = retail_dataframe()
    issues = run_quality_checks(df, infer_schema(df))
    checks = {(issue["check"], issue["column"]) for issue in issues}

    assert ("negative_values", "days_to_ship") in checks
    assert ("imbalanced", "Returned") in checks
    assert ("constant_column", "Comment") in checks


def test_quality_flags_inconsistent_labels():
    df = pd.DataFrame({"City": ["Pune", "pune ", "Mumbai", "Delhi"] * 10})
    issues = run_quality_checks(df, infer_schema(df))

    assert any(issue["check"] == "inconsistent_labels" for issue in issues)


def test_name_helpers():
    assert name_tokens("CustomerID") == ["customer", "id"]
    assert name_tokens("days_since_order") == ["days", "since", "order"]
    assert humanize("OrderRate") == "Order Rate"
    assert humanize("days_to_ship") == "Days To Ship"
