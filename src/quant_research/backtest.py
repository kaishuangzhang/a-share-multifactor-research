from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class Position:
    shares: int
    entry_date: pd.Timestamp


def run_backtests(
    panel: pd.DataFrame,
    targets: pd.DataFrame,
    config: dict[str, Any],
    output_dir: Path,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    if targets.empty:
        raise ValueError("No rebalance targets to backtest")

    equity_curves: dict[str, pd.DataFrame] = {}
    trade_logs: list[pd.DataFrame] = []
    for strategy, strategy_targets in targets.groupby("strategy", sort=True):
        equity, trades = run_single_backtest(panel, strategy_targets, strategy, config)
        equity_curves[strategy] = equity
        trade_logs.append(trades)

    output_dir.mkdir(parents=True, exist_ok=True)
    for strategy, equity in equity_curves.items():
        equity.to_parquet(output_dir / f"equity_{strategy}.parquet", index=False)
    trades = pd.concat(trade_logs, ignore_index=True)
    trades.to_parquet(output_dir / "trade_log.parquet", index=False)
    return equity_curves, trades


def run_single_backtest(
    panel: pd.DataFrame,
    targets: pd.DataFrame,
    strategy: str,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    settings = config["backtest"]
    initial_cash = float(settings["initial_cash"])
    commission_rate = float(settings["commission_rate"])
    minimum_commission = float(settings["minimum_commission"])
    stamp_tax_rate = float(settings["stamp_tax_rate"])
    stamp_tax_legacy_rate = float(settings.get("stamp_tax_legacy_rate", stamp_tax_rate))
    stamp_tax_cut_date = pd.Timestamp(
        settings.get("stamp_tax_cut_date", "1900-01-01")
    )
    slippage_rate = float(settings["slippage_rate"])
    lot_size = int(settings["lot_size"])
    max_adv_participation = float(settings["max_adv_participation"])

    required = {
        "date",
        "code",
        "open",
        "close",
        "valuation_close",
        "prev_close",
        "amount",
        "tradeable",
    }
    missing = required.difference(panel.columns)
    if missing:
        raise ValueError(f"Panel missing backtest columns: {sorted(missing)}")

    market_dates = pd.DatetimeIndex(sorted(panel["date"].dropna().unique()))
    start_date = pd.Timestamp(targets["trade_date"].min())
    end_date = pd.Timestamp(config["splits"]["test_end"])
    dates = market_dates[(market_dates >= start_date) & (market_dates <= end_date)]
    day_lookup = {
        date: frame.set_index("code", drop=False)
        for date, frame in panel.groupby("date", sort=False)
    }
    target_lookup = {
        date: frame.set_index("code")
        for date, frame in targets.groupby("trade_date", sort=False)
    }

    cash = initial_cash
    positions: dict[str, Position] = {}
    last_prices: dict[str, float] = {}
    equity_rows: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []

    for date in dates:
        daily = day_lookup.get(pd.Timestamp(date))
        if daily is None:
            continue

        if pd.Timestamp(date) in target_lookup:
            target_day = target_lookup[pd.Timestamp(date)]
            weights = target_day["weight"].to_dict()
            signal_dates = target_day["signal_date"].to_dict()
            effective_stamp_tax_rate = (
                stamp_tax_rate
                if pd.Timestamp(date) >= stamp_tax_cut_date
                else stamp_tax_legacy_rate
            )
            cash, trade_rows = _rebalance(
                strategy=strategy,
                date=pd.Timestamp(date),
                daily=daily,
                weights=weights,
                signal_dates=signal_dates,
                cash=cash,
                positions=positions,
                last_prices=last_prices,
                trade_rows=trade_rows,
                commission_rate=commission_rate,
                minimum_commission=minimum_commission,
                stamp_tax_rate=effective_stamp_tax_rate,
                slippage_rate=slippage_rate,
                lot_size=lot_size,
                max_adv_participation=max_adv_participation,
            )

        total_value = float(cash)
        for code, position in positions.items():
            if code not in daily.index:
                price = last_prices.get(code, 0.0)
            else:
                row = daily.loc[code]
                price = row["close"] if row["close"] > 0 else row["valuation_close"]
                if pd.notna(price) and price > 0:
                    last_prices[code] = float(price)
                else:
                    price = last_prices.get(code, 0.0)
            total_value += position.shares * float(price)

        gross_exposure = (
            sum(
                position.shares * last_prices.get(code, 0.0)
                for code, position in positions.items()
            )
            / total_value
            if total_value > 0
            else 0.0
        )
        equity_rows.append(
            {
                "date": pd.Timestamp(date),
                "strategy": strategy,
                "total_value": total_value,
                "cash": cash,
                "num_positions": len(positions),
                "gross_exposure": gross_exposure,
            }
        )

    equity = pd.DataFrame(equity_rows).sort_values("date").reset_index(drop=True)
    equity["daily_ret"] = equity["total_value"].pct_change(fill_method=None)
    equity.loc[equity.index[0], "daily_ret"] = (
        equity.loc[equity.index[0], "total_value"] / initial_cash - 1.0
    )
    equity["nav"] = equity["total_value"] / initial_cash
    equity["cum_max"] = equity["nav"].cummax()
    equity["drawdown"] = equity["nav"] / equity["cum_max"] - 1.0
    trades = pd.DataFrame(trade_rows)
    return equity, trades


def _rebalance(
    *,
    strategy: str,
    date: pd.Timestamp,
    daily: pd.DataFrame,
    weights: dict[str, float],
    signal_dates: dict[str, pd.Timestamp],
    cash: float,
    positions: dict[str, Position],
    last_prices: dict[str, float],
    trade_rows: list[dict[str, Any]],
    commission_rate: float,
    minimum_commission: float,
    stamp_tax_rate: float,
    slippage_rate: float,
    lot_size: int,
    max_adv_participation: float,
) -> tuple[float, list[dict[str, Any]]]:
    def reference_price(code: str) -> float:
        if code not in daily.index:
            return last_prices.get(code, 0.0)
        row = daily.loc[code]
        if row["open"] > 0 and row["tradeable"]:
            return float(row["open"])
        return float(last_prices.get(code, row["valuation_close"]))

    total_before = float(cash) + sum(
        position.shares * reference_price(code)
        for code, position in positions.items()
    )

    # Reduce existing positions first.
    for code in list(positions.keys()):
        if code not in daily.index:
            continue
        row = daily.loc[code]
        if not _can_trade(row, date, "SELL"):
            continue
        raw_price = float(row["open"])
        desired_weight = float(weights.get(code, 0.0))
        desired_shares = (
            int((total_before * desired_weight) / raw_price / lot_size) * lot_size
            if desired_weight > 0
            else 0
        )
        sell_shares = positions[code].shares - desired_shares
        if sell_shares < lot_size:
            continue
        adv_shares = int(
            (float(row["amount"]) * max_adv_participation) / raw_price / lot_size
        ) * lot_size
        sell_shares = min(sell_shares, adv_shares)
        if sell_shares < lot_size:
            continue

        execution_price = raw_price * (1.0 - slippage_rate)
        notional = sell_shares * execution_price
        commission = max(minimum_commission, notional * commission_rate)
        stamp_tax = notional * stamp_tax_rate
        cash += notional - commission - stamp_tax
        positions[code].shares -= sell_shares
        if positions[code].shares == 0:
            del positions[code]

        trade_rows.append(
            _trade_record(
                strategy,
                signal_dates.get(code, pd.NaT),
                date,
                code,
                "SELL",
                sell_shares,
                raw_price,
                execution_price,
                notional,
                commission,
                stamp_tax,
                cash,
            )
        )

    total_after_sells = float(cash) + sum(
        position.shares * reference_price(code)
        for code, position in positions.items()
    )

    # Increase underweight target positions second.
    for code, weight in sorted(weights.items(), key=lambda item: item[1], reverse=True):
        if code not in daily.index:
            continue
        row = daily.loc[code]
        if not _can_trade(row, date, "BUY"):
            continue
        raw_price = float(row["open"])
        current_shares = positions.get(code, Position(0, date)).shares
        desired_shares = (
            int((total_after_sells * float(weight)) / raw_price / lot_size) * lot_size
        )
        buy_shares = desired_shares - current_shares
        if buy_shares < lot_size:
            continue

        adv_shares = int(
            (float(row["amount"]) * max_adv_participation) / raw_price / lot_size
        ) * lot_size
        buy_shares = min(buy_shares, adv_shares)
        buy_shares = _shares_affordable(
            buy_shares,
            cash,
            raw_price * (1.0 + slippage_rate),
            commission_rate,
            minimum_commission,
            lot_size,
        )
        if buy_shares < lot_size:
            continue

        execution_price = raw_price * (1.0 + slippage_rate)
        notional = buy_shares * execution_price
        commission = max(minimum_commission, notional * commission_rate)
        total_cost = notional + commission
        if total_cost > cash:
            continue

        cash -= total_cost
        if code in positions:
            positions[code].shares += buy_shares
        else:
            positions[code] = Position(buy_shares, date)
        last_prices[code] = raw_price
        trade_rows.append(
            _trade_record(
                strategy,
                signal_dates.get(code, pd.NaT),
                date,
                code,
                "BUY",
                buy_shares,
                raw_price,
                execution_price,
                notional,
                commission,
                0.0,
                cash,
            )
        )

    return cash, trade_rows


def _can_trade(row: pd.Series, date: pd.Timestamp, side: str) -> bool:
    if not bool(row["tradeable"]):
        return False
    prev_close = row["prev_close"]
    if pd.isna(prev_close) or prev_close <= 0:
        return True

    limit_pct = _price_limit_pct(str(row["code"]), date)
    tolerance = 0.011
    open_price = float(row["open"])
    if side == "BUY":
        return open_price < prev_close * (1.0 + limit_pct) - tolerance
    return open_price > prev_close * (1.0 - limit_pct) + tolerance


def _price_limit_pct(code: str, date: pd.Timestamp) -> float:
    if code.startswith(("688", "689")):
        return 0.20
    if code.startswith(("300", "301")):
        return 0.20 if date >= pd.Timestamp("2020-08-24") else 0.10
    return 0.10


def _shares_affordable(
    requested_shares: int,
    cash: float,
    execution_price: float,
    commission_rate: float,
    minimum_commission: float,
    lot_size: int,
) -> int:
    shares = requested_shares
    while shares >= lot_size:
        notional = shares * execution_price
        commission = max(minimum_commission, notional * commission_rate)
        if notional + commission <= cash:
            return shares
        shares -= lot_size
    return 0


def _trade_record(
    strategy: str,
    signal_date: pd.Timestamp,
    trade_date: pd.Timestamp,
    code: str,
    side: str,
    shares: int,
    reference_price: float,
    execution_price: float,
    notional: float,
    commission: float,
    stamp_tax: float,
    cash_after: float,
) -> dict[str, Any]:
    return {
        "strategy": strategy,
        "signal_date": signal_date,
        "trade_date": trade_date,
        "code": code,
        "side": side,
        "shares": shares,
        "reference_price": reference_price,
        "execution_price": execution_price,
        "notional": notional,
        "commission": commission,
        "stamp_tax": stamp_tax,
        "slippage_cost": abs(execution_price - reference_price) * shares,
        "cash_after": cash_after,
    }

