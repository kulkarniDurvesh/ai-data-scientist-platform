import pandas as pd

from core.schema_inference import DatasetSchema, infer_schema


def profile_column(df, column):
    s = df[column]

    total = len(s)
    non_null = s.dropna()
    unique = s.nunique(dropna=True)
    missing = s.isna().sum()

    result = {
        "column": column,
        "dtype": str(s.dtype),
        "rows": total,
        "non_null": len(non_null),
        "missing": missing,
        "missing_pct": missing / total * 100 if total else 0,
        "unique": unique,
        "unique_pct": unique / total * 100 if total else 0,
        "is_numeric": pd.api.types.is_numeric_dtype(s),
        "is_datetime": pd.api.types.is_datetime64_any_dtype(s),
        "is_boolean": pd.api.types.is_bool_dtype(s),
    }

    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        result.update({
            "min": non_null.min() if len(non_null) else None,
            "max": non_null.max() if len(non_null) else None,
            "mean": non_null.mean() if len(non_null) else None,
            "median": non_null.median() if len(non_null) else None,
            "std": non_null.std() if len(non_null) else None,
            "skewness": non_null.skew() if len(non_null) else None,
            "zero_pct": (
                (non_null == 0).mean() * 100
                if len(non_null) else None
            ),
            "negative_pct": (
                (non_null < 0).mean() * 100
                if len(non_null) else None
            )
        })

    return result


def profile_dataset(df, schema: DatasetSchema | None = None):

    schema = schema or infer_schema(df)

    # Step 1: Create profile for every column
    columns = [
        profile_column(df, column)
        for column in df.columns
    ]

    # Step 2: Create complete dataset profile
    profile = {
        "rows": len(df),
        "columns": len(df.columns),
        "duplicate_rows": int(df.duplicated().sum()),
        "column_profiles": columns,
        "primary_time": schema.primary_time,
    }

    # Step 3: Semantic role comes from the shared schema.
    # High-cardinality dimensions are excluded from automatic
    # group comparisons, so they get their own role here.
    for column in profile["column_profiles"]:
        column_schema = schema.columns[str(column["column"])]

        role = column_schema.role

        if role == "dimension" and column_schema.high_cardinality:
            role = "dimension_high"

        column["semantic_role"] = role

    return profile
