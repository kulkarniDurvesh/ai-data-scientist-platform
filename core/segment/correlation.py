"""
How features relate to each other.

    numeric pairs      Spearman rank correlation (robust to skew and outliers)
    redundant pairs    |rho| >= 0.9: one of the two adds little; keep one
    VIF                variance inflation factor: > 10 means a column is
                       largely explained by the others (multicollinearity)
    category pairs     Cramér's V between categorical columns

Correlation is association, not causation: the report says which
columns move together, not why.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

REDUNDANT = 0.9
MAX_NUMERIC = 25
HIGH_VIF = 10


@dataclass
class CorrelationReport:
    matrix: pd.DataFrame              # Spearman, numeric x numeric
    pairs: pd.DataFrame               # strongest numeric pairs
    redundant: pd.DataFrame           # |rho| >= REDUNDANT, with a suggestion
    vif: pd.DataFrame | None
    categorical: pd.DataFrame         # Cramér's V pairs


def correlation_report(frame: pd.DataFrame, numeric: list[str], categorical: list[str]) -> CorrelationReport:
    numeric = [c for c in numeric if frame[c].nunique() > 1][:MAX_NUMERIC]
    data = frame[numeric].apply(pd.to_numeric, errors="coerce")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        matrix = data.corr(method="spearman") if len(numeric) > 1 else pd.DataFrame()

    rows = []
    for i, a in enumerate(numeric):
        for b in numeric[i + 1:]:
            rho = matrix.loc[a, b] if not matrix.empty else np.nan
            if not pd.isna(rho):
                rows.append({"Feature A": a, "Feature B": b, "Spearman": round(float(rho), 3)})

    pairs = pd.DataFrame(rows, columns=["Feature A", "Feature B", "Spearman"])
    if not pairs.empty:
        pairs = pairs.reindex(pairs["Spearman"].abs().sort_values(ascending=False).index).reset_index(drop=True)

    redundant = pairs[pairs["Spearman"].abs() >= REDUNDANT].copy()
    if not redundant.empty:
        redundant["Suggestion"] = [
            f"Keep one: dropping {b} loses little information"
            for b in redundant["Feature B"]
        ]

    return CorrelationReport(
        matrix=matrix.round(3),
        pairs=pairs.head(15),
        redundant=redundant.reset_index(drop=True),
        vif=_vif(data),
        categorical=_cramers_pairs(frame, categorical),
    )


def _vif(data: pd.DataFrame) -> pd.DataFrame | None:
    clean = data.dropna()
    if clean.shape[1] < 3 or len(clean) < clean.shape[1] * 5:
        return None

    standardized = (clean - clean.mean()) / clean.std(ddof=0).replace(0, np.nan)
    standardized = standardized.dropna(axis=1)
    if standardized.shape[1] < 3:
        return None

    values = standardized.to_numpy()
    rows = []
    for index, column in enumerate(standardized.columns):
        others = np.delete(values, index, axis=1)
        target = values[:, index]
        coefficients, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(others)), others]), target, rcond=None)
        fitted = np.column_stack([np.ones(len(others)), others]) @ coefficients
        r2 = 1 - np.sum((target - fitted) ** 2) / np.sum((target - target.mean()) ** 2)
        vif = float("inf") if r2 >= 0.9999 else 1 / (1 - r2)
        rows.append({"Feature": column, "VIF": round(vif, 1) if math.isfinite(vif) else vif})

    table = pd.DataFrame(rows).sort_values("VIF", ascending=False).reset_index(drop=True)
    exact = ~np.isfinite(table["VIF"].astype(float))
    table["Note"] = np.select(
        [exact, table["VIF"].astype(float) > HIGH_VIF],
        ["an exact combination of other columns (e.g. a total of parts)", "largely explained by other columns"],
        default="",
    )
    table["VIF"] = [("∞" if not np.isfinite(v) else v) for v in table["VIF"].astype(float)]
    return table


def _cramers_pairs(frame: pd.DataFrame, categorical: list[str]) -> pd.DataFrame:
    rows = []
    for i, a in enumerate(categorical):
        for b in categorical[i + 1:]:
            table = pd.crosstab(frame[a], frame[b])
            if min(table.shape) < 2:
                continue
            chi2, p_value, _, _ = stats.chi2_contingency(table)
            n = table.to_numpy().sum()
            v = math.sqrt(chi2 / (n * (min(table.shape) - 1))) if n else 0.0
            rows.append({"Feature A": a, "Feature B": b, "Cramér's V": round(v, 3), "p-value": float(f"{p_value:.3g}")})

    result = pd.DataFrame(rows, columns=["Feature A", "Feature B", "Cramér's V", "p-value"])
    return result.sort_values("Cramér's V", ascending=False).reset_index(drop=True)
