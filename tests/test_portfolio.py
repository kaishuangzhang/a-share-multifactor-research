from __future__ import annotations

import pandas as pd
import pytest

from quant_research.portfolio import build_monthly_targets


def test_targets_trade_on_the_next_market_date(tmp_path) -> None:
    dates = pd.bdate_range("2023-01-02", periods=45)
    rows = []
    for date in dates:
        for index in range(25):
            rows.append(
                {
                    "date": date,
                    "code": f"{index:06d}",
                    "eligible": True,
                    "baseline_score": float(index),
                    "model_score": float(index),
                }
            )
    scored = pd.DataFrame(rows)
    config = {
        "splits": {"test_start": "2023-01-01", "test_end": "2023-03-31"},
        "portfolio": {"top_n": 10, "min_stocks": 5, "gross_exposure": 0.95},
    }
    targets = build_monthly_targets(scored, config, tmp_path)
    assert not targets.empty
    for (strategy, signal_date), group in targets.groupby(["strategy", "signal_date"]):
        expected = dates[dates > signal_date][0]
        assert group["trade_date"].unique().tolist() == [expected]
        assert group["weight"].sum() == pytest.approx(0.95)
