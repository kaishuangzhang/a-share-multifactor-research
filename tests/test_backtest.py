from __future__ import annotations

import pandas as pd
import pytest

from quant_research.backtest import run_single_backtest


def test_backtest_does_not_trade_on_signal_date(tmp_path) -> None:
    dates = pd.bdate_range("2023-01-02", periods=8)
    rows = []
    for code in ["000001", "000002"]:
        for date in dates:
            rows.append(
                {
                    "date": date,
                    "code": code,
                    "open": 10.0,
                    "close": 10.0,
                    "valuation_close": 10.0,
                    "prev_close": 10.0,
                    "amount": 1_000_000_000.0,
                    "tradeable": True,
                }
            )
    panel = pd.DataFrame(rows)
    signal_date = dates[2]
    trade_date = dates[3]
    targets = pd.DataFrame(
        {
            "strategy": ["baseline", "baseline"],
            "signal_date": [signal_date, signal_date],
            "trade_date": [trade_date, trade_date],
            "code": ["000001", "000002"],
            "score": [2.0, 1.0],
            "weight": [0.475, 0.475],
        }
    )
    config = {
        "splits": {"test_end": dates[-1].date().isoformat()},
        "backtest": {
            "initial_cash": 1000000.0,
            "commission_rate": 0.0003,
            "minimum_commission": 5.0,
            "stamp_tax_rate": 0.0005,
            "stamp_tax_legacy_rate": 0.001,
            "stamp_tax_cut_date": "2023-08-28",
            "slippage_rate": 0.001,
            "lot_size": 100,
            "max_adv_participation": 0.05,
        },
    }
    _, trades = run_single_backtest(panel, targets, "baseline", config)
    assert not trades.empty
    assert trades["trade_date"].min() == trade_date
    assert trades["signal_date"].max() < trades["trade_date"].min()
    assert trades["cash_after"].min() >= 0
    assert trades["notional"].sum() > 0
    assert "stamp_tax" in trades.columns
