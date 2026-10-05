"""
Which groups need attention?

    unusual     each group's value in the current period against its own
                history (robust z: median and MAD of earlier periods)
    attention   a ranking of groups combining
                  change vs the comparison period (%)
                  how unusual the current period is for the group
                  recent trend (slope over the last periods, % per period)
                  size (share of the total), so big groups count more
                with the factors that drove each score listed

"Bad" is a fall when higher is better, a rise when lower is better.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.forecast.series import ROW_COUNT
from core.schema_inference import format_number, humanize

from .change import ChangeSpec, choose_periods, complete_periods, periods, value

HISTORY_PERIODS = 12
TREND_PERIODS = 6
UNUSUAL_Z = 2.5


def group_history(frame: pd.DataFrame, spec: ChangeSpec, dimension: str) -> pd.DataFrame:
    """Value per group and period (groups x periods)."""

    stamps = complete_periods(frame, spec)
    stamp = periods(frame, spec)
    data = frame[stamp.isin(stamps)].assign(_period=stamp)

    table = data.groupby([dimension, "_period"]).apply(lambda part: value(part, spec), include_groups=False).unstack("_period")
    table = table.reindex(columns=stamps)
    if spec.aggregation in {"sum", "count"} or spec.measure == ROW_COUNT:
        table = table.fillna(0.0)
    return table


def attention_ranking(
    frame: pd.DataFrame,
    spec: ChangeSpec,
    dimension: str,
    higher_is_better: bool = True,
    top: int = 20,
) -> pd.DataFrame:

    current, previous = choose_periods(frame, spec)
    history = group_history(frame, spec, dimension)
    if current not in history.columns or previous not in history.columns:
        return pd.DataFrame()

    position = list(history.columns).index(current)
    past = history.iloc[:, max(0, position - HISTORY_PERIODS):position]
    recent = history.iloc[:, max(0, position - TREND_PERIODS + 1):position + 1]
    total_now = history[current].abs().sum()
    sign = 1 if higher_is_better else -1

    rows = []
    for group, series in history.iterrows():
        now, before = series[current], series[previous]
        change_pct = (now - before) / abs(before) if before else (np.nan if now == 0 else np.sign(now))

        median = past.loc[group].median()
        mad = (past.loc[group] - median).abs().median()
        scale = 1.4826 * mad if mad > 0 else past.loc[group].std(ddof=0)
        z = (now - median) / scale if scale and scale > 0 else 0.0

        level = recent.loc[group].abs().mean()
        slope = np.polyfit(np.arange(recent.shape[1]), recent.loc[group].to_numpy(dtype=float), 1)[0] if recent.shape[1] >= 3 else 0.0
        trend_pct = slope / level if level else 0.0

        share = abs(now) / total_now if total_now else 0.0

        # Only the "bad" direction raises attention; each factor capped.
        decline = max(0.0, -sign * (change_pct if not pd.isna(change_pct) else 0.0))
        unusual = max(0.0, -sign * z)
        worsening = max(0.0, -sign * trend_pct)
        score = (
            min(decline, 1.0) * 40
            + min(unusual / 4, 1.0) * 30
            + min(worsening * 5, 1.0) * 20
        ) * (0.5 + 0.5 * min(share * len(history), 1.0)) + min(share, 1.0) * 10

        factors = []
        if decline > 0.05:
            factors.append(f"{'down' if higher_is_better else 'up'} {abs(change_pct):.0%} vs previous")
        if unusual >= UNUSUAL_Z:
            factors.append(f"unusually {'low' if higher_is_better else 'high'} for this group (z {z:+.1f})")
        if worsening > 0.02:
            factors.append(f"trend {'falling' if higher_is_better else 'rising'} {abs(trend_pct):.0%} per period")
        if share >= 1.5 / max(len(history), 1):
            factors.append(f"large share of total ({share:.0%})")

        rows.append({
            dimension: group,
            "Attention score": round(score, 1),
            "Previous": round(float(before), 2),
            "Current": round(float(now), 2),
            "Change %": round(float(change_pct) * 100, 1) if not pd.isna(change_pct) else None,
            "Unusual (z)": round(float(z), 2),
            "Trend % / period": round(float(trend_pct) * 100, 1),
            "Why": "; ".join(factors) or "no warning signs",
        })

    table = pd.DataFrame(rows).sort_values("Attention score", ascending=False)
    return table.head(top).reset_index(drop=True)


def unusual_groups(frame: pd.DataFrame, spec: ChangeSpec, dimension: str) -> pd.DataFrame:
    """Groups whose current value is far from their own history (|z| >= 2.5)."""

    current, _ = choose_periods(frame, spec)
    history = group_history(frame, spec, dimension)
    if current not in history.columns:
        return pd.DataFrame()

    position = list(history.columns).index(current)
    past = history.iloc[:, max(0, position - HISTORY_PERIODS):position]
    if past.shape[1] < 4:
        return pd.DataFrame()

    rows = []
    for group, series in history.iterrows():
        now = series[current]
        median = past.loc[group].median()
        mad = (past.loc[group] - median).abs().median()
        scale = 1.4826 * mad if mad > 0 else past.loc[group].std(ddof=0)
        if not scale or scale <= 0:
            continue
        z = (now - median) / scale
        if abs(z) >= UNUSUAL_Z:
            low, high = past.loc[group].quantile(0.25), past.loc[group].quantile(0.75)
            rows.append({
                humanize(dimension): group,
                "Current": round(float(now), 2),
                "Usual range": f"{format_number(low)}–{format_number(high)}",
                "z": round(float(z), 2),
                "Direction": "above usual" if z > 0 else "below usual",
            })

    result = pd.DataFrame(rows)
    return result.reindex(result["z"].abs().sort_values(ascending=False).index).reset_index(drop=True) if len(result) else result
