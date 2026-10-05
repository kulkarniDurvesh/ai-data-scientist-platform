"""
Segments with readable profiles.

    prepare   numbers: median impute, log for strongly skewed positives,
              standardise; categories: one-hot, weighted so each category
              column counts about as much as one number
    k-means   k = 2..8, chosen by silhouette (or fixed by the user)
    bands     simple alternative: low / medium / high thirds of one measure
    profile   size, share, and what makes each segment different from the
              rest (standardised differences, over-represented categories)
    name      "High Order Rate, Low Not Interested", from the profile
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from core.schema_inference import format_number, humanize

RANDOM_STATE = 42
K_RANGE = range(2, 9)
SILHOUETTE_SAMPLE = 3000
DISTINCT_Z = 0.5          # standardised difference that counts as a trait
DISTINCT_SHARE = 0.2      # extra share of a category that counts as a trait


@dataclass
class Clustering:
    labels: np.ndarray
    names: dict[int, str]
    k: int
    silhouette: float | None
    k_scores: pd.DataFrame | None     # k, silhouette, inertia
    profiles: pd.DataFrame            # one row per segment
    heatmap: pd.DataFrame             # segment x numeric feature, standardised difference
    projection: pd.DataFrame          # PC1, PC2, segment (sampled)
    method: str


def prepare_matrix(frame: pd.DataFrame, numeric: list[str], categorical: list[str]) -> np.ndarray:
    blocks = []

    for column in numeric:
        values = pd.to_numeric(frame[column], errors="coerce").astype(float)
        values = values.fillna(values.median() if values.notna().any() else 0.0)
        if values.min() >= 0 and values.skew() > 1:
            values = np.log1p(values)
        std = values.std(ddof=0)
        blocks.append(((values - values.mean()) / std if std > 0 else values * 0).to_numpy()[:, None])

    for column in categorical:
        dummies = pd.get_dummies(frame[column].astype(str), dtype=float)
        if dummies.shape[1] > 1:
            blocks.append(dummies.to_numpy() / np.sqrt(dummies.shape[1]))

    if not blocks:
        raise ValueError("No usable features to segment on.")

    return np.hstack(blocks)


def kmeans_segments(
    frame: pd.DataFrame,
    numeric: list[str],
    categorical: list[str],
    k: int | None = None,
) -> Clustering:

    matrix = prepare_matrix(frame, numeric, categorical)
    n = len(matrix)
    if n < 10:
        raise ValueError("At least 10 units are needed to find segments.")

    sample = _sample(n)
    rows = []
    fits = {}

    for candidate in K_RANGE:
        if candidate >= n:
            break
        model = KMeans(n_clusters=candidate, n_init=10, random_state=RANDOM_STATE).fit(matrix)
        fits[candidate] = model
        score = silhouette_score(matrix[sample], model.labels_[sample]) if len(set(model.labels_[sample])) > 1 else np.nan
        rows.append({"k": candidate, "Silhouette": round(float(score), 3), "Inertia": round(float(model.inertia_), 1)})

    k_scores = pd.DataFrame(rows)
    chosen = k if k and k in fits else int(k_scores.loc[k_scores["Silhouette"].idxmax(), "k"])
    labels = fits[chosen].labels_
    silhouette = float(k_scores.loc[k_scores["k"] == chosen, "Silhouette"].iloc[0])

    labels = _order_by_size(labels)
    return _finish(frame, matrix, labels, numeric, categorical, silhouette, k_scores, "K-means")


def band_segments(frame: pd.DataFrame, measure: str, numeric: list[str], categorical: list[str]) -> Clustering:
    values = pd.to_numeric(frame[measure], errors="coerce")
    ranks = values.rank(method="first")
    bands = pd.qcut(ranks, 3, labels=False)
    labels = bands.fillna(0).astype(int).to_numpy()

    matrix = prepare_matrix(frame, numeric, categorical)
    names = {0: f"Low {humanize(measure)}", 1: f"Medium {humanize(measure)}", 2: f"High {humanize(measure)}"}
    result = _finish(frame, matrix, labels, numeric, categorical, None, None, f"Bands of {humanize(measure)}")
    result.names = names
    result.profiles["Segment"] = result.profiles["Id"].map(names)
    return result


def _finish(frame, matrix, labels, numeric, categorical, silhouette, k_scores, method) -> Clustering:
    profiles, heatmap, names = _profiles(frame, labels, numeric, categorical)

    projection = None
    if matrix.shape[1] >= 2:
        sample = _sample(len(matrix), 2000)
        points = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(matrix)
        projection = pd.DataFrame({
            "PC1": points[sample, 0], "PC2": points[sample, 1],
            "Segment": [names[label] for label in labels[sample]],
        })

    return Clustering(
        labels=labels, names=names, k=len(names), silhouette=silhouette, k_scores=k_scores,
        profiles=profiles, heatmap=heatmap,
        projection=projection if projection is not None else pd.DataFrame(), method=method,
    )


def _profiles(frame, labels, numeric, categorical):
    data = frame.copy()
    data["_segment"] = labels

    means = data[numeric].apply(pd.to_numeric, errors="coerce").mean() if numeric else pd.Series(dtype=float)
    stds = data[numeric].apply(pd.to_numeric, errors="coerce").std(ddof=0).replace(0, np.nan) if numeric else pd.Series(dtype=float)

    rows, heat, names = [], {}, {}
    for segment, part in data.groupby("_segment"):
        traits = []

        diffs = {}
        for column in numeric:
            value = pd.to_numeric(part[column], errors="coerce").mean()
            z = (value - means[column]) / stds[column] if not pd.isna(stds[column]) else 0.0
            diffs[column] = 0.0 if pd.isna(z) else float(z)
        heat[segment] = diffs

        for column, z in sorted(diffs.items(), key=lambda item: -abs(item[1])):
            if abs(z) >= DISTINCT_Z:
                word = "High" if z > 0 else "Low"
                traits.append((abs(z), f"{word} {humanize(column)}", f"{word.lower()} {humanize(column)} ({format_number(pd.to_numeric(part[column], errors='coerce').mean())} vs {format_number(means[column])})"))

        for column in categorical:
            share = part[column].value_counts(normalize=True)
            overall = data[column].value_counts(normalize=True)
            if share.empty:
                continue
            top = share.index[0]
            lift = share.iloc[0] - overall.get(top, 0)
            if lift >= DISTINCT_SHARE:
                traits.append((lift * 2, f"Mostly {top}", f"{share.iloc[0]:.0%} {humanize(column)} = {top} (vs {overall.get(top, 0):.0%} overall)"))

        traits.sort(key=lambda item: -item[0])
        name = ", ".join(t[1] for t in traits[:2]) or "Typical"
        names[segment] = name

        rows.append({
            "Id": segment,
            "Segment": name,
            "Units": len(part),
            "Share": round(len(part) / len(data), 3),
            "What sets it apart": "; ".join(t[2] for t in traits[:4]) or "close to average on every feature",
        })

    names = _unique(names)
    profiles = pd.DataFrame(rows)
    profiles["Segment"] = profiles["Id"].map(names)

    heatmap = pd.DataFrame(heat).T
    heatmap.index = [names[i] for i in heatmap.index]
    return profiles, heatmap.round(2), names


def _unique(names: dict[int, str]) -> dict[int, str]:
    seen: dict[str, int] = {}
    result = {}
    for key, name in names.items():
        seen[name] = seen.get(name, 0) + 1
        result[key] = name if seen[name] == 1 else f"{name} ({seen[name]})"
    return result


def _order_by_size(labels: np.ndarray) -> np.ndarray:
    order = pd.Series(labels).value_counts().index
    mapping = {old: new for new, old in enumerate(order)}
    return np.array([mapping[label] for label in labels])


def _sample(n: int, size: int = SILHOUETTE_SAMPLE) -> np.ndarray:
    if n <= size:
        return np.arange(n)
    return np.sort(np.random.default_rng(RANDOM_STATE).choice(n, size, replace=False))
