"""
Why did a number change?

    compare     a measure in the latest complete period vs the previous
                period (or the same period a year earlier)
    decompose   each group's contribution to the change, for one dimension:
                  sums / counts  contribution = current - previous per group
                  averages       mix effect  (the groups' weights changed)
                                 rate effect (values within groups changed)
    drill down  follow the group that explains most of the change into the
                next dimension that concentrates it best, a few levels deep

Everything is computed from the data; the narrative only puts numbers
that were calculated here into words.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from core.forecast.series import ROW_COUNT, period_offset
from core.schema_inference import format_number, humanize
from core.target_analysis import native_grain

MISSING = "(missing)"

MAX_DEPTH = 3
# A group must explain at least this share of the change to be followed.
MIN_SHARE_TO_FOLLOW = 0.25
MAX_GROUPS_SHOWN = 10


@dataclass
class ChangeSpec:
    time: str
    measure: str                       # column or ROW_COUNT
    aggregation: str = "sum"           # sum | count | mean
    freq: str = "M"
    compare: str = "previous"          # previous | year
    dimensions: list[str] = field(default_factory=list)
    period: pd.Timestamp | None = None # None = latest complete period


@dataclass
class Comparison:
    current_period: pd.Timestamp
    previous_period: pd.Timestamp
    current: float
    previous: float

    @property
    def change(self) -> float:
        return self.current - self.previous

    @property
    def change_pct(self) -> float:
        return self.change / abs(self.previous) if self.previous else float("nan")


@dataclass
class Breakdown:
    dimension: str
    table: pd.DataFrame                # group, previous, current, change, share (+ mix / rate)
    concentration: float               # share of the change explained by the top group


@dataclass
class Step:
    path: list[tuple[str, object]]     # filters that lead to this step
    comparison: Comparison
    breakdown: Breakdown | None
    text: str


def periods(frame: pd.DataFrame, spec: ChangeSpec) -> pd.Series:
    return frame[spec.time].dt.to_period(spec.freq).dt.start_time


def complete_periods(frame: pd.DataFrame, spec: ChangeSpec) -> list[pd.Timestamp]:
    """Distinct periods, leaving out a last period the data only partly covers."""

    stamps = sorted(periods(frame, spec).dropna().unique())
    if not stamps:
        return []

    # Snapshot columns (one value per period) are complete by definition.
    if native_grain(frame[spec.time]) == spec.freq:
        return [pd.Timestamp(stamp) for stamp in stamps]

    last = pd.Timestamp(stamps[-1])
    following = last + pd.tseries.frequencies.to_offset(period_offset(spec.freq))
    covered = (frame[spec.time].max().normalize() - last).days + 1
    if covered / max((following - last).days, 1) < 0.8 and len(stamps) > 2:
        stamps = stamps[:-1]
    return [pd.Timestamp(stamp) for stamp in stamps]


def choose_periods(frame: pd.DataFrame, spec: ChangeSpec) -> tuple[pd.Timestamp, pd.Timestamp]:
    stamps = complete_periods(frame, spec)
    if len(stamps) < 2:
        raise ValueError("At least two complete periods are needed to explain a change.")

    current = pd.Timestamp(spec.period).to_period(spec.freq).start_time if spec.period is not None else stamps[-1]
    if current not in stamps:
        raise ValueError(f"No data for the period starting {current:%Y-%m-%d}.")

    if spec.compare == "year":
        previous = current - pd.DateOffset(years=1)
        previous = previous.to_period(spec.freq).start_time
        if previous not in stamps:
            raise ValueError("There is no data for the same period one year earlier.")
    else:
        position = stamps.index(current)
        if position == 0:
            raise ValueError("The chosen period is the first one; there is nothing to compare with.")
        previous = stamps[position - 1]

    return current, previous


def value(frame: pd.DataFrame, spec: ChangeSpec) -> float:
    if frame.empty:
        return 0.0
    if spec.measure == ROW_COUNT or spec.aggregation == "count":
        return float(len(frame))
    values = pd.to_numeric(frame[spec.measure], errors="coerce")
    return float(values.mean()) if spec.aggregation == "mean" else float(values.sum())


def compare(frame: pd.DataFrame, spec: ChangeSpec, current: pd.Timestamp, previous: pd.Timestamp) -> Comparison:
    stamp = periods(frame, spec)
    return Comparison(
        current_period=current,
        previous_period=previous,
        current=value(frame[stamp == current], spec),
        previous=value(frame[stamp == previous], spec),
    )


def breakdown(
    frame: pd.DataFrame,
    spec: ChangeSpec,
    dimension: str,
    current: pd.Timestamp,
    previous: pd.Timestamp,
) -> Breakdown:

    stamp = periods(frame, spec)
    # Missing values form their own group so contributions add up to the total.
    frame = frame.assign(**{dimension: frame[dimension].astype(object).where(frame[dimension].notna(), MISSING)})
    now, before = frame[stamp == current], frame[stamp == previous]
    keys = sorted(set(now[dimension].dropna().unique()) | set(before[dimension].dropna().unique()), key=str)

    if spec.aggregation == "mean" and spec.measure != ROW_COUNT:
        table = _mix_rate(now, before, spec, dimension, keys)
    else:
        rows = []
        for key in keys:
            a = value(before[before[dimension] == key], spec)
            b = value(now[now[dimension] == key], spec)
            rows.append({dimension: key, "Previous": a, "Current": b, "Change": b - a})
        table = pd.DataFrame(rows)

    total = table["Change"].sum()
    gross = table["Change"].abs().sum()
    table["Share of change"] = table["Change"] / total if total else 0.0
    table = table.reindex(table["Change"].abs().sort_values(ascending=False).index).reset_index(drop=True)

    # How much of all movement the biggest group moving with the total
    # accounts for (0..1); unlike the share of the net change, it can't
    # exceed 100% when groups offset each other.
    same_direction = table[np.sign(table["Change"]) == np.sign(total)] if total else table
    concentration = float(abs(same_direction["Change"].iloc[0]) / gross) if len(same_direction) and gross else 0.0

    return Breakdown(dimension, table, concentration)


def _mix_rate(now, before, spec, dimension, keys) -> pd.DataFrame:
    """Shift-share for averages: total change = mix effect + rate effect."""

    values_now = pd.to_numeric(now[spec.measure], errors="coerce")
    values_before = pd.to_numeric(before[spec.measure], errors="coerce")
    n_now, n_before = len(now), len(before)

    rows = []
    for key in keys:
        in_now = now[dimension] == key
        in_before = before[dimension] == key
        w1 = in_now.sum() / n_now if n_now else 0.0
        w0 = in_before.sum() / n_before if n_before else 0.0
        r1 = float(values_now[in_now].mean()) if in_now.any() else np.nan
        r0 = float(values_before[in_before].mean()) if in_before.any() else np.nan
        mix = (w1 - w0) * (r0 if not np.isnan(r0) else (r1 if not np.isnan(r1) else 0.0))
        rate = w1 * ((r1 - r0) if not (np.isnan(r1) or np.isnan(r0)) else 0.0)
        rows.append({
            dimension: key, "Previous": r0, "Current": r1,
            "Weight before": w0, "Weight now": w1,
            "Mix effect": mix, "Rate effect": rate, "Change": mix + rate,
        })
    return pd.DataFrame(rows)


def explain_change(frame: pd.DataFrame, spec: ChangeSpec) -> list[Step]:
    """
    The explanation chain: overall comparison, then at each level the
    dimension where one group explains most of the change, followed into
    that group, up to MAX_DEPTH levels.
    """

    current, previous = choose_periods(frame, spec)
    steps: list[Step] = []
    path: list[tuple[str, object]] = []
    subset = frame
    remaining = [d for d in spec.dimensions if d in frame.columns]

    for depth in range(MAX_DEPTH + 1):
        comparison = compare(subset, spec, current, previous)
        if not remaining or comparison.change == 0:
            steps.append(Step(list(path), comparison, None, _text(spec, path, comparison, None)))
            break

        candidates = [breakdown(subset, spec, d, current, previous) for d in remaining]
        candidates = [c for c in candidates if len(c.table) > 1]
        if not candidates:
            steps.append(Step(list(path), comparison, None, _text(spec, path, comparison, None)))
            break

        best = max(candidates, key=lambda c: c.concentration)
        text = _text(spec, path, comparison, best)
        if best.concentration < MIN_SHARE_TO_FOLLOW:
            text += (
                f" No single group stands out (the largest accounts for "
                f"{best.concentration:.0%} of all movement across {humanize(best.dimension)}), "
                f"so the change is spread out and the drill-down stops here."
            )
        steps.append(Step(list(path), comparison, best, text))

        if depth == MAX_DEPTH or best.concentration < MIN_SHARE_TO_FOLLOW:
            break

        top = best.table[np.sign(best.table["Change"]) == np.sign(comparison.change)].iloc[0]
        key = top[best.dimension]
        path.append((best.dimension, key))
        subset = subset[subset[best.dimension] == key]
        remaining = [d for d in remaining if d != best.dimension]

    return steps


def all_breakdowns(frame: pd.DataFrame, spec: ChangeSpec) -> list[Breakdown]:
    current, previous = choose_periods(frame, spec)
    found = [breakdown(frame, spec, d, current, previous) for d in spec.dimensions if d in frame.columns]
    return sorted([b for b in found if len(b.table) > 1], key=lambda b: -b.concentration)


def measure_label(spec: ChangeSpec) -> str:
    if spec.measure == ROW_COUNT:
        return "Number of rows"
    name = humanize(spec.measure)
    prefix = {"sum": "Total", "mean": "Average"}.get(spec.aggregation, "")
    return name if name.lower().startswith(prefix.lower()) else f"{prefix} {name}".strip()


def _text(spec: ChangeSpec, path, comparison: Comparison, best: Breakdown | None) -> str:
    where = " and ".join(f"{humanize(d)} = {k}" for d, k in path)
    scope = f"Within {where}, " if where else ""
    direction = "rose" if comparison.change > 0 else "fell" if comparison.change < 0 else "did not change"
    pct = f" ({comparison.change_pct:+.1%})" if not np.isnan(comparison.change_pct) else ""
    label = measure_label(spec)
    text = (
        f"{scope}{label.lower() if scope else label} {direction} from "
        f"{format_number(comparison.previous)} to {format_number(comparison.current)}{pct} "
        f"between {_period(comparison.previous_period, spec)} and {_period(comparison.current_period, spec)}."
    )
    if best is not None and comparison.change != 0:
        top = best.table[np.sign(best.table["Change"]) == np.sign(comparison.change)]
        if len(top):
            row = top.iloc[0]
            text += f" The biggest contributor is {humanize(best.dimension)} = {row[best.dimension]} ({format_number(row['Change'])}"
            if row["Share of change"] <= 1:
                text += f", {row['Share of change']:.0%} of the change)."
            else:
                against = best.table[np.sign(best.table["Change"]) == -np.sign(comparison.change)]
                text += ", more than the whole net change"
                if len(against):
                    other = against.iloc[0]
                    text += (
                        f": other groups moved the other way, the largest being "
                        f"{other[best.dimension]} ({format_number(other['Change'])})"
                    )
                text += ")."

            if "Mix effect" in best.table.columns:
                mix, rate = best.table["Mix effect"].sum(), best.table["Rate effect"].sum()
                text += (
                    f" Across {humanize(best.dimension)}, {format_number(rate)} comes from values "
                    f"changing within groups and {format_number(mix)} from the mix of groups shifting."
                )
    return text


def _period(stamp: pd.Timestamp, spec: ChangeSpec) -> str:
    if spec.freq == "M":
        return f"{stamp:%b %Y}"
    if spec.freq == "Q":
        return f"Q{stamp.quarter} {stamp.year}"
    if spec.freq == "W":
        return f"week of {stamp:%Y-%m-%d}"
    return f"{stamp:%Y-%m-%d}"
