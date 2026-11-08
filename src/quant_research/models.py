from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor

from .features import FACTOR_COLUMNS


FEATURE_COLUMNS = [f"z_{factor}" for factor in FACTOR_COLUMNS]


def fit_baseline_and_model(
    data: pd.DataFrame,
    train_mask: pd.Series,
    validation_mask: pd.Series,
    config: dict[str, Any],
    output_dir: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    train = data.loc[
        train_mask
        & data["eligible"]
        & data["target_rank"].notna()
        & data[FEATURE_COLUMNS].notna().all(axis=1)
    ].copy()
    if len(train) < 1000:
        raise ValueError(f"Not enough training rows after filtering: {len(train)}")

    model_config = config["model"]
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        max_iter=int(model_config.get("max_iter", 200)),
        learning_rate=float(model_config.get("learning_rate", 0.04)),
        max_leaf_nodes=int(model_config.get("max_leaf_nodes", 15)),
        min_samples_leaf=int(model_config.get("min_samples_leaf", 100)),
        l2_regularization=float(model_config.get("l2_regularization", 2.0)),
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
        random_state=int(model_config.get("random_state", 42)),
    )
    model.fit(train[FEATURE_COLUMNS], train["target_rank"])

    scored = data.copy()
    score_mask = scored["eligible"] & scored[FEATURE_COLUMNS].notna().all(axis=1)
    scored["baseline_score"] = np.nan
    scored["model_score"] = np.nan
    scored.loc[score_mask, "baseline_score"] = scored.loc[
        score_mask, FEATURE_COLUMNS
    ].mean(axis=1)
    scored.loc[score_mask, "model_score"] = model.predict(
        scored.loc[score_mask, FEATURE_COLUMNS]
    )

    validation = scored.loc[
        validation_mask
        & scored["eligible"]
        & scored["target_excess_return"].notna()
        & scored["baseline_score"].notna(),
        ["date", "baseline_score", "model_score", "target_excess_return"],
    ]
    validation_summary = {
        "model": _score_rank_ic_summary(validation, "model_score"),
        "baseline": _score_rank_ic_summary(validation, "baseline_score"),
        "train_rows": len(train),
        "train_start": scored.loc[train_mask, "date"].min().date().isoformat(),
        "train_end": scored.loc[train_mask, "date"].max().date().isoformat(),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    score_columns = [
        "date",
        "code",
        "eligible",
        "baseline_score",
        "model_score",
        "target_excess_return",
        "target_rank",
        "label_end_date",
    ]
    scored[score_columns].to_parquet(output_dir / "model_scores.parquet", index=False)
    with (output_dir / "model_validation.json").open("w", encoding="utf-8") as handle:
        json.dump(validation_summary, handle, ensure_ascii=False, indent=2)
    return scored, validation_summary


def _score_rank_ic_summary(data: pd.DataFrame, score_column: str) -> dict[str, float]:
    daily_ic = []
    for _, group in data.groupby("date", sort=True):
        valid = group[[score_column, "target_excess_return"]].dropna()
        if (
            len(valid) >= 20
            and valid[score_column].nunique() >= 2
            and valid["target_excess_return"].nunique() >= 2
        ):
            value = spearmanr(valid[score_column], valid["target_excess_return"]).statistic
            if np.isfinite(value):
                daily_ic.append(float(value))
    series = pd.Series(daily_ic, dtype=float)
    return {
        "mean_rank_ic": float(series.mean()) if len(series) else float("nan"),
        "std_rank_ic": float(series.std()) if len(series) else float("nan"),
        "icir": (
            float(series.mean() / series.std())
            if len(series) and series.std() > 0
            else float("nan")
        ),
        "positive_ratio": float((series > 0).mean()) if len(series) else float("nan"),
        "days": int(len(series)),
    }

