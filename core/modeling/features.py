"""
Generic feature preparation from column roles.

    numeric  (measures, 0/1 columns) -> impute median (+ scale for linear models)
    category (text, yes/no as text)   -> fill "Missing" -> one-hot, rare levels grouped

The same steps run at training and at scoring time because they live
inside the scikit-learn pipeline.
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from core.schema_inference import DatasetSchema

# Category levels rarer than this share are grouped as "infrequent".
MIN_CATEGORY_SHARE = 0.01

MISSING = "Missing"


def split_columns(schema: DatasetSchema, features: list[str]) -> tuple[list[str], list[str]]:
    numeric = [column for column in features if schema.is_numeric(column)]
    categorical = [column for column in features if column not in numeric]
    return numeric, categorical


def prepare_frame(
    dataframe: pd.DataFrame,
    numeric: list[str],
    categorical: list[str],
) -> pd.DataFrame:
    """Model input: numbers as floats, categories as plain text."""

    frame = pd.DataFrame(index=dataframe.index)

    for column in numeric:
        frame[column] = pd.to_numeric(dataframe[column], errors="coerce").astype(float)

    for column in categorical:
        values = dataframe[column].astype(object)
        frame[column] = values.where(values.notna(), MISSING).astype(str)

    return frame


def build_preprocessor(
    numeric: list[str],
    categorical: list[str],
    scale: bool,
) -> ColumnTransformer:

    numeric_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))

    transformers = []

    if numeric:
        transformers.append(("numeric", Pipeline(numeric_steps), numeric))

    if categorical:
        transformers.append((
            "categorical",
            OneHotEncoder(
                handle_unknown="infrequent_if_exist",
                min_frequency=MIN_CATEGORY_SHARE,
                sparse_output=False,
            ),
            categorical,
        ))

    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)
