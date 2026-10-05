"""
Compute KPIs from a domain configuration.

    tables (by name) + DomainConfig
        -> each KPI per period (and optionally per group role)
        -> value, numerator, denominator, definition, rows used, timestamp

KPIs whose columns or roles are missing are reported as unavailable
with the reason; they never stop the others. Nothing is evaluated as
code: only the fixed operations in core.kpi.config are supported.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable

import numpy as np
import pandas as pd

from core.forecast.series import period_offset
from core.target_analysis import native_grain

from .config import Aggregate, ConfigError, DomainConfig, KpiDef

Tables = Callable[[str | None], "pd.DataFrame | None"]


@dataclass
class KpiReport:
    summary: pd.DataFrame        # one row per KPI: current, previous, change, status
    by_group: pd.DataFrame | None
    trend: pd.DataFrame          # KPI x period
    unavailable: pd.DataFrame    # KPI, reason
    current_period: pd.Timestamp | None
    previous_period: pd.Timestamp | None
    computed_at: str


def evaluate_domain(
    config: DomainConfig,
    tables: Tables,
    freq: str = "M",
    group_role: str | None = None,
    periods_back: int = 12,
) -> KpiReport:

    computed_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows, unavailable, trend_rows, group_frames = [], [], [], []

    all_periods: set[pd.Timestamp] = set()
    prepared: list[tuple[KpiDef, dict]] = []

    for kpi in config.kpis:
        try:
            parts = {"numerator": _prepare(kpi, kpi.numerator, config, tables, freq, group_role)}
            if kpi.denominator is not None:
                parts["denominator"] = _prepare(kpi, kpi.denominator, config, tables, freq, group_role)
            prepared.append((kpi, parts))
            for part in parts.values():
                if part["periods"] is not None:
                    all_periods |= set(part["periods"].dropna().unique())
        except (ConfigError, KeyError, ValueError, TypeError) as error:
            unavailable.append({"KPI": kpi.label, "Reason": str(error.args[0]) if error.args else str(error)})

    ordered = sorted(pd.Timestamp(p) for p in all_periods)
    ordered = _drop_partial_last(ordered, prepared, freq)
    current = ordered[-1] if ordered else None
    previous = ordered[-2] if len(ordered) > 1 else None
    window = ordered[-periods_back:]

    for kpi, parts in prepared:
        series = {period: _value(parts, period) for period in window} if window else {}
        overall = _value(parts, None) if not window else None
        now = series.get(current, overall) if current is not None else overall
        before = series.get(previous) if previous is not None else None
        n = _rows(parts, current)

        change = None if now is None or before is None else now - before
        status = ""
        if change is not None and change != 0:
            good = (change > 0) == (kpi.direction == "up")
            status = "better" if good else "worse"

        rows.append({
            "KPI": kpi.label,
            "Current": now,
            "Previous": before,
            "Change": change,
            "Status": status,
            "Format": kpi.format,
            "Direction": "higher is better" if kpi.direction == "up" else "lower is better",
            "Definition": kpi.definition(),
            "Rows used": n,
            "Description": kpi.description,
            "_id": kpi.id,
        })
        for period, value in series.items():
            trend_rows.append({"KPI": kpi.label, "Period": period, "Value": value})

        if group_role:
            frame = _by_group(kpi, parts, current)
            if frame is not None:
                group_frames.append(frame)

    by_group = None
    if group_role and group_frames:
        by_group = pd.concat(group_frames, axis=1).reset_index()
        by_group = by_group.rename(columns={by_group.columns[0]: group_role.replace("_", " ").title()})

    return KpiReport(
        summary=pd.DataFrame(rows),
        by_group=by_group,
        trend=pd.DataFrame(trend_rows, columns=["KPI", "Period", "Value"]),
        unavailable=pd.DataFrame(unavailable, columns=["KPI", "Reason"]),
        current_period=current,
        previous_period=previous,
        computed_at=computed_at,
    )


def _prepare(kpi: KpiDef, aggregate: Aggregate, config: DomainConfig, tables: Tables, freq: str, group_role: str | None) -> dict:
    """Filtered rows for one aggregate, with their period and group keys."""

    table = aggregate.table or kpi.table
    frame = tables(table)
    if frame is None:
        raise ConfigError(f"table '{table}' not found")

    mask = pd.Series(True, index=frame.index)
    for key, condition in aggregate.where.items():
        column = config.column(key, table)
        if column not in frame.columns:
            raise ConfigError(f"column '{column}' (for {key}) not in table '{table or 'current'}'")
        mask &= _condition(frame[column], condition)
    rows = frame[mask]

    target = None
    if aggregate.target is not None:
        target = config.column(aggregate.target, table)
        if target not in frame.columns:
            raise ConfigError(f"column '{target}' (for {aggregate.target}) not in table '{table or 'current'}'")

    periods = None
    if config.has_role("date", table):
        date = config.column("date", table)
        if date not in frame.columns:
            raise ConfigError(f"date column '{date}' not in table '{table or 'current'}'")
        dates = pd.to_datetime(frame[date], errors="coerce")
        periods = dates[mask].dt.to_period(freq).dt.start_time
        last_date, native = dates.max(), native_grain(dates) == freq

    groups = None
    if group_role:
        if not config.has_role(group_role, table):
            raise ConfigError(f"role '{group_role}' has no column in table '{table or 'current'}'")
        group_column = config.column(group_role, table)
        if group_column not in frame.columns:
            raise ConfigError(f"column '{group_column}' (for {group_role}) not in table '{table or 'current'}'")
        groups = rows[group_column]

    return {
        "aggregate": aggregate, "rows": rows, "target": target, "periods": periods, "groups": groups,
        "last_date": last_date if periods is not None else None,
        "native": native if periods is not None else True,
    }


def _drop_partial_last(ordered: list, prepared: list, freq: str) -> list:
    """Leave out a last period that event data covers less than 80% of."""

    if len(ordered) < 3:
        return ordered
    parts = [part for _, parts in prepared for part in parts.values() if part["periods"] is not None]
    if not parts or all(part["native"] for part in parts):
        return ordered

    last = ordered[-1]
    following = last + pd.tseries.frequencies.to_offset(period_offset(freq))
    latest = max(part["last_date"] for part in parts if not pd.isna(part["last_date"]))
    covered = (latest.normalize() - last).days + 1
    return ordered[:-1] if covered / max((following - last).days, 1) < 0.8 else ordered


def _condition(series: pd.Series, condition) -> pd.Series:
    if isinstance(condition, list):
        return series.isin(condition)
    if isinstance(condition, dict):
        mask = pd.Series(True, index=series.index)
        numbers = pd.to_numeric(series, errors="coerce")
        for op, value in condition.items():
            if op == "gt":
                mask &= numbers > value
            elif op == "ge":
                mask &= numbers >= value
            elif op == "lt":
                mask &= numbers < value
            elif op == "le":
                mask &= numbers <= value
            elif op == "ne":
                mask &= series != value
            elif op == "eq":
                mask &= series == value
        return mask
    return series == condition


def _aggregate(part: dict, rows: pd.DataFrame) -> float:
    op, target = part["aggregate"].op, part["target"]
    if op == "count":
        return float(len(rows)) if target is None else float(rows[target].notna().sum())
    if op == "distinct":
        return float(rows[target].nunique())
    values = pd.to_numeric(rows[target], errors="coerce")
    if values.dropna().empty:
        return float("nan")
    return float(getattr(values, op)())


def _select(part: dict, period, group=None) -> pd.DataFrame:
    rows = part["rows"]
    mask = pd.Series(True, index=rows.index)
    if period is not None and part["periods"] is not None:
        mask &= part["periods"] == period
    if group is not None and part["groups"] is not None:
        mask &= part["groups"] == group
    return rows[mask]


def _value(parts: dict, period, group=None) -> float | None:
    numerator = _aggregate(parts["numerator"], _select(parts["numerator"], period, group))
    if "denominator" not in parts:
        return numerator
    denominator = _aggregate(parts["denominator"], _select(parts["denominator"], period, group))
    return numerator / denominator if denominator else None


def _rows(parts: dict, period) -> int:
    return int(len(_select(parts["numerator"], period)))


def _by_group(kpi: KpiDef, parts: dict, period) -> pd.DataFrame | None:
    groups = parts["numerator"]["groups"]
    if groups is None:
        return None
    values = {
        group: _value(parts, period, group)
        for group in sorted(groups.dropna().unique(), key=str)
    }
    return pd.Series(values, name=kpi.label).to_frame()


def format_value(value, fmt: str) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "–"
    if fmt == "percent":
        return f"{value:.1%}"
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    return f"{value:,.2f}".rstrip("0").rstrip(".")
