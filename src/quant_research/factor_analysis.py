from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .features import FACTOR_COLUMNS


def run_factor_analysis(
    data: pd.DataFrame,
    research_mask: pd.Series,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sample = data.loc[
        research_mask
        & data["eligible"]
        & data["target_excess_return"].notna(),
        ["date", "code", *FACTOR_COLUMNS, "target_excess_return"],
    ].copy()

    daily_rows: list[dict[str, Any]] = []
    for date, day in sample.groupby("date", sort=True):
        if len(day) < 20:
            continue
        for factor in FACTOR_COLUMNS:
            valid = day[[factor, "target_excess_return"]].dropna()
            if (
                len(valid) < 20
                or valid[factor].nunique() < 2
                or valid["target_excess_return"].nunique() < 2
            ):
                continue
            ic = spearmanr(valid[factor], valid["target_excess_return"]).statistic
            if np.isfinite(ic):
                daily_rows.append(
                    {"date": date, "factor": factor, "rank_ic": float(ic), "n": len(valid)}
                )

    daily_ic = pd.DataFrame(daily_rows)
    summary_rows = []
    for factor, group in daily_ic.groupby("factor", sort=False):
        ic_values = group["rank_ic"]
        summary_rows.append(
            {
                "factor": factor,
                "mean_rank_ic": ic_values.mean(),
                "std_rank_ic": ic_values.std(),
                "icir": (
                    ic_values.mean() / ic_values.std()
                    if ic_values.std() > 0
                    else np.nan
                ),
                "positive_ic_ratio": (ic_values > 0).mean(),
                "median_observations": group["n"].median(),
                "days": len(group),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values("mean_rank_ic", ascending=False)

    decile_rows: list[dict[str, Any]] = []
    for date, day in sample.groupby("date", sort=True):
        if len(day) < 50:
            continue
        for factor in FACTOR_COLUMNS:
            valid = day[[factor, "target_excess_return"]].dropna()
            if len(valid) < 50 or valid[factor].nunique() < 5:
                continue
            try:
                valid = valid.assign(
                    decile=pd.qcut(
                        valid[factor].rank(method="first"),
                        5,
                        labels=False,
                    )
                    + 1
                )
            except ValueError:
                continue
            for decile, group in valid.groupby("decile"):
                decile_rows.append(
                    {
                        "date": date,
                        "factor": factor,
                        "bucket": int(decile),
                        "future_excess_return": group["target_excess_return"].mean(),
                    }
                )

    decile_daily = pd.DataFrame(decile_rows)
    decile_summary = (
        decile_daily.groupby(["factor", "bucket"], as_index=False)[
            "future_excess_return"
        ]
        .mean()
        .sort_values(["factor", "bucket"])
    )

    correlation_rows: list[dict[str, float | str]] = []
    for _, day in sample.groupby("date", sort=True):
        valid = day[FACTOR_COLUMNS].dropna()
        if len(valid) < 20:
            continue
        correlations = valid.corr(method="spearman")
        for left_index, left in enumerate(FACTOR_COLUMNS):
            for right in FACTOR_COLUMNS[left_index + 1 :]:
                value = correlations.loc[left, right]
                if pd.notna(value):
                    correlation_rows.append(
                        {
                            "factor_left": left,
                            "factor_right": right,
                            "correlation": float(value),
                        }
                    )
    correlation = (
        pd.DataFrame(correlation_rows)
        .groupby(["factor_left", "factor_right"], as_index=False)["correlation"]
        .mean()
        if correlation_rows
        else pd.DataFrame(columns=["factor_left", "factor_right", "correlation"])
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    daily_ic.to_parquet(output_dir / "factor_rank_ic_daily.parquet", index=False)
    summary.to_csv(output_dir / "factor_rank_ic_summary.csv", index=False)
    decile_summary.to_csv(output_dir / "factor_bucket_returns.csv", index=False)
    correlation.to_csv(output_dir / "factor_correlation.csv", index=False)
    return daily_ic, summary, decile_summary, correlation
