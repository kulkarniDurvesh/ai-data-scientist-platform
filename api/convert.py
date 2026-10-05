"""Turn analysis results into plain JSON-safe values."""

from __future__ import annotations

import math
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd


def plain(value: Any) -> Any:
    """A JSON-safe version of one value (NaN -> None, timestamps -> ISO text)."""

    if value is None:
        return None
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return None if pd.isna(value) else value.isoformat()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return None if math.isnan(value) or math.isinf(value) else value
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def table(frame: pd.DataFrame | None, limit: int | None = None, offset: int = 0) -> dict | None:
    """{columns, rows, total_rows} for a dataframe, optionally one page of it."""

    if frame is None:
        return None
    total = len(frame)
    page = frame.iloc[offset: offset + limit] if limit is not None else frame.iloc[offset:]
    columns = [str(column) for column in page.columns]
    rows = [
        {column: plain(value) for column, value in zip(columns, record)}
        for record in page.itertuples(index=False, name=None)
    ]
    return {"columns": columns, "rows": rows, "total_rows": total}
