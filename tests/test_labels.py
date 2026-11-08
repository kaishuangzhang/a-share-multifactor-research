from __future__ import annotations

import pandas as pd
import pytest

from quant_research.labels import add_forward_labels


def test_label_starts_at_next_open_and_ends_after_horizon() -> None:
    dates = pd.bdate_range("2024-01-02", periods=24)
    panel = pd.DataFrame(
        {
            "date": dates,
            "code": "000001",
            "open": [100.0 + i for i in range(24)],
        }
    )
    config = {"labels": {"horizon_days": 20}}
    result = add_forward_labels(panel, config)

    assert result.loc[0, "entry_open"] == pytest.approx(101.0)
    assert result.loc[0, "exit_open"] == pytest.approx(121.0)
    assert result.loc[0, "future_return"] == pytest.approx(121.0 / 101.0 - 1.0)
    assert result.loc[0, "label_end_date"] == dates[21]


def test_training_label_cannot_cross_validation_period() -> None:
    dates = pd.bdate_range("2021-12-01", periods=45)
    panel = pd.DataFrame(
        {
            "date": dates,
            "code": "000001",
            "open": [10.0 + i for i in range(45)],
        }
    )
    config = {"labels": {"horizon_days": 20}}
    result = add_forward_labels(panel, config)
    eligible = result.loc[
        result["date"].le("2021-12-31") & result["label_end_date"].lt("2022-01-01")
    ]
    assert eligible["label_end_date"].max() < pd.Timestamp("2022-01-01")


def test_missing_or_zero_execution_price_produces_no_label() -> None:
    dates = pd.bdate_range("2024-01-02", periods=24)
    opens = [100.0 + i for i in range(24)]
    opens[1] = 0.0
    panel = pd.DataFrame({"date": dates, "code": "000001", "open": opens})
    result = add_forward_labels(panel, {"labels": {"horizon_days": 20}})
    assert pd.isna(result.loc[0, "future_return"])

