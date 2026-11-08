from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def build_monthly_targets(
    scored: pd.DataFrame,
    config: dict[str, Any],
    output_dir: Path,
) -> pd.DataFrame:
    start = pd.Timestamp(config["splits"]["test_start"])
    end = pd.Timestamp(config["splits"]["test_end"])
    top_n = int(config["portfolio"]["top_n"])
    min_stocks = int(config["portfolio"]["min_stocks"])
    gross_exposure = float(config["portfolio"]["gross_exposure"])

    market_dates = pd.DatetimeIndex(sorted(scored["date"].dropna().unique()))
    in_test = market_dates[(market_dates >= start) & (market_dates <= end)]
    if len(in_test) == 0:
        raise ValueError("No market dates in test period")

    monthly_last = (
        pd.Series(in_test, index=in_test)
        .groupby(in_test.to_period("M"))
        .max()
        .sort_values()
        .tolist()
    )
    next_trade_date = {
        date: market_dates[index + 1]
        for index, date in enumerate(market_dates[:-1])
    }

    records: list[pd.DataFrame] = []
    for strategy, score_column in [
        ("baseline", "baseline_score"),
        ("model", "model_score"),
    ]:
        for signal_date in monthly_last:
            if pd.Timestamp(signal_date) not in next_trade_date:
                continue
            day = scored.loc[
                scored["date"].eq(signal_date)
                & scored["eligible"]
                & scored[score_column].notna(),
                ["date", "code", score_column],
            ].copy()
            if len(day) < min_stocks:
                continue
            selected = day.nlargest(top_n, score_column).copy()
            selected["strategy"] = strategy
            selected["signal_date"] = signal_date
            selected["trade_date"] = next_trade_date[pd.Timestamp(signal_date)]
            selected["weight"] = gross_exposure / len(selected)
            records.append(
                selected.rename(columns={score_column: "score"})[
                    ["strategy", "signal_date", "trade_date", "code", "score", "weight"]
                ]
            )

    targets = pd.concat(records, ignore_index=True) if records else pd.DataFrame()
    output_dir.mkdir(parents=True, exist_ok=True)
    targets.to_parquet(output_dir / "rebalance_targets.parquet", index=False)
    return targets

