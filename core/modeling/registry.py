"""
Local model registry.

    models/<name>/model.joblib   fitted pipeline + feature lists
    models/<name>/meta.json      goal, metrics, data fingerprint, timestamp

Saved models can be listed, reloaded and used to score new data with the
same columns. Phase 5 serves them over HTTP.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from .builder import ModelResult, result_summary_dict

DEFAULT_FOLDER = Path(__file__).resolve().parents[2] / "models"


def save_model(
    result: ModelResult,
    dataframe: pd.DataFrame,
    dataset_name: str,
    folder: Path | str = DEFAULT_FOLDER,
) -> dict[str, Any]:

    created = datetime.now(timezone.utc)
    slug = re.sub(r"[^A-Za-z0-9]+", "-", f"{result.spec.target}-{result.spec.goal_type}").strip("-").lower()
    name = f"{slug}-{created.strftime('%Y%m%d-%H%M%S')}"

    target = Path(folder) / name
    target.mkdir(parents=True, exist_ok=False)

    meta = {
        "name": name,
        "created": created.isoformat(timespec="seconds"),
        "dataset": dataset_name,
        "rows": int(len(dataframe)),
        "fingerprint": fingerprint(dataframe),
        **result_summary_dict(result),
    }

    joblib.dump(
        {"pipeline": result.pipeline, "numeric": result.numeric,
         "categorical": result.categorical, "task": result.task},
        target / "model.joblib",
    )
    (target / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return meta


def list_models(folder: Path | str = DEFAULT_FOLDER) -> list[dict[str, Any]]:
    folder = Path(folder)
    if not folder.exists():
        return []

    metas = []
    for path in folder.glob("*/meta.json"):
        try:
            metas.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue

    return sorted(metas, key=lambda meta: meta.get("created", ""), reverse=True)


def load_model(name: str, folder: Path | str = DEFAULT_FOLDER) -> dict[str, Any]:
    path = Path(folder) / name
    bundle = joblib.load(path / "model.joblib")
    bundle["meta"] = json.loads((path / "meta.json").read_text(encoding="utf-8"))
    return bundle


def fingerprint(dataframe: pd.DataFrame) -> str:
    """Identifies the data a model was trained on (columns, size, sample)."""

    text = "|".join(map(str, dataframe.columns)) + f"|{len(dataframe)}|"
    text += pd.util.hash_pandas_object(dataframe.head(500), index=False).astype(str).str.cat()
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
