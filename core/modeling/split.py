"""
Honest train / validation / test splits.

    dated data   -> by time: test = latest ~20% of periods,
                    validation = latest ~20% of the remaining training periods
    undated data -> random (stratified for yes/no targets)

Model selection uses the validation part only; the test part is touched
once, to report how the chosen model does on unseen (later) data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

TEST_SHARE = 0.2
VALIDATION_SHARE = 0.2
RANDOM_STATE = 42


@dataclass
class Split:
    method: str                  # time | stratified | random
    train: pd.Index              # model fitting for the reported test score
    test: pd.Index
    inner_train: pd.Index        # model selection
    validation: pd.Index
    description: str


def make_split(
    dataframe: pd.DataFrame,
    y: pd.Series,
    time_column: str | None,
    stratify: bool,
) -> Split:

    labelled = y.index[y.notna()]

    if time_column and dataframe.loc[labelled, time_column].nunique() >= 3:
        return _time_split(dataframe, labelled, time_column)

    return _random_split(labelled, y, stratify)


def _time_split(dataframe: pd.DataFrame, labelled: pd.Index, time_column: str) -> Split:
    times = dataframe.loc[labelled, time_column]
    periods = np.sort(times.dropna().unique())

    test_from = periods[_cut(len(periods), TEST_SHARE)]
    train = labelled[(times < test_from).to_numpy()]
    test = labelled[(times >= test_from).to_numpy()]

    train_periods = np.sort(dataframe.loc[train, time_column].dropna().unique())
    valid_from = train_periods[_cut(len(train_periods), VALIDATION_SHARE)]
    train_times = dataframe.loc[train, time_column]
    inner_train = train[(train_times < valid_from).to_numpy()]
    validation = train[(train_times >= valid_from).to_numpy()]

    description = (
        f"By time on {time_column}: train before {_fmt(test_from)} "
        f"({len(train):,} rows), test from {_fmt(test_from)} ({len(test):,} rows). "
        f"Models are compared on a validation window from {_fmt(valid_from)}, "
        f"then the chosen one is scored once on the test period."
    )

    return Split("time", train, test, inner_train, validation, description)


def _random_split(labelled: pd.Index, y: pd.Series, stratify: bool) -> Split:
    rng = np.random.default_rng(RANDOM_STATE)

    def divide(index: pd.Index, share: float) -> tuple[pd.Index, pd.Index]:
        if stratify:
            parts_a, parts_b = [], []
            for _, group in y.loc[index].groupby(y.loc[index]):
                shuffled = rng.permutation(group.index.to_numpy())
                n_b = max(1, round(len(shuffled) * share))
                parts_b.append(shuffled[:n_b])
                parts_a.append(shuffled[n_b:])
            return pd.Index(np.concatenate(parts_a)), pd.Index(np.concatenate(parts_b))
        shuffled = rng.permutation(index.to_numpy())
        n_b = max(1, round(len(shuffled) * share))
        return pd.Index(shuffled[n_b:]), pd.Index(shuffled[:n_b])

    train, test = divide(labelled, TEST_SHARE)
    inner_train, validation = divide(train, VALIDATION_SHARE)

    method = "stratified" if stratify else "random"
    description = (
        f"{'Stratified r' if stratify else 'R'}andom split (no date column): "
        f"train {len(train):,} rows, test {len(test):,} rows; models compared "
        f"on a validation part of the training rows."
    )

    return Split(method, train, test, inner_train, validation, description)


def _cut(n: int, share: float) -> int:
    """
    Index of the first held-out period: the first ceil(n x (1 - share))
    periods are kept for fitting (the same cut the Target tab recommends).
    """
    return min(n - 1, max(1, math.ceil(n * (1 - share))))


def _fmt(value) -> str:
    try:
        return pd.Timestamp(value).strftime("%Y-%m-%d")
    except (TypeError, ValueError):
        return str(value)
