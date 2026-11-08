from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def evaluate_strategies(
    equity_curves: dict[str, pd.DataFrame],
    benchmark: pd.DataFrame,
    trades: pd.DataFrame,
    config: dict[str, Any],
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    initial_cash = float(config["backtest"]["initial_cash"])
    result_rows: list[dict[str, Any]] = []
    yearly_rows: list[dict[str, Any]] = []

    for strategy, equity in equity_curves.items():
        merged = equity.merge(
            benchmark[["date", "benchmark_ret", "benchmark_nav"]],
            on="date",
            how="left",
        )
        if merged["benchmark_ret"].isna().all():
            merged["benchmark_ret"] = 0.0
        merged["benchmark_ret"] = merged["benchmark_ret"].fillna(0.0)
        merged["benchmark_rebased"] = (1.0 + merged["benchmark_ret"]).cumprod()
        merged["active_nav"] = merged["nav"] / merged["benchmark_rebased"]

        returns = merged["daily_ret"].dropna()
        excess = returns - merged.loc[returns.index, "benchmark_ret"]
        years = len(merged) / 252.0
        total_return = merged["nav"].iloc[-1] - 1.0
        annual_return = (
            (1.0 + total_return) ** (1.0 / years) - 1.0 if years > 0 else np.nan
        )
        annual_volatility = returns.std() * np.sqrt(252)
        max_drawdown = merged["drawdown"].min()
        sharpe = (
            returns.mean() / returns.std() * np.sqrt(252)
            if returns.std() > 0
            else np.nan
        )
        calmar = annual_return / abs(max_drawdown) if max_drawdown < 0 else np.nan
        tracking_error = excess.std() * np.sqrt(252)
        information_ratio = (
            excess.mean() / excess.std() * np.sqrt(252)
            if excess.std() > 0
            else np.nan
        )
        benchmark_return = merged["benchmark_rebased"].iloc[-1] - 1.0
        active_return = merged["active_nav"].iloc[-1] - 1.0

        strategy_trades = trades.loc[trades["strategy"].eq(strategy)]
        buy_notional = strategy_trades.loc[
            strategy_trades["side"].eq("BUY"), "notional"
        ].sum()
        average_equity = merged["total_value"].mean()
        annual_turnover = (
            buy_notional / average_equity / years if average_equity > 0 and years > 0 else np.nan
        )
        total_cost = (
            strategy_trades["commission"].sum()
            + strategy_trades["stamp_tax"].sum()
            + strategy_trades["slippage_cost"].sum()
        )
        result_rows.append(
            {
                "strategy": strategy,
                "initial_cash": initial_cash,
                "final_value": merged["total_value"].iloc[-1],
                "total_return": total_return,
                "annual_return": annual_return,
                "annual_volatility": annual_volatility,
                "max_drawdown": max_drawdown,
                "sharpe": sharpe,
                "calmar": calmar,
                "daily_win_rate": (returns > 0).mean(),
                "benchmark_return": benchmark_return,
                "active_return": active_return,
                "tracking_error": tracking_error,
                "information_ratio": information_ratio,
                "annual_one_way_turnover": annual_turnover,
                "total_explicit_cost": total_cost,
                "days": len(merged),
            }
        )

        yearly = merged.assign(year=merged["date"].dt.year).groupby("year")
        for year, group in yearly:
            yearly_rows.append(
                {
                    "strategy": strategy,
                    "year": int(year),
                    "return": group["nav"].iloc[-1] / group["nav"].iloc[0] - 1.0,
                    "max_drawdown": group["drawdown"].min(),
                }
            )

    summary = pd.DataFrame(result_rows)
    yearly_summary = pd.DataFrame(yearly_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "performance_summary.csv", index=False)
    yearly_summary.to_csv(output_dir / "yearly_performance.csv", index=False)
    with (output_dir / "performance_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(result_rows, handle, ensure_ascii=False, indent=2, default=str)
    return summary, yearly_summary
