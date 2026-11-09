from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


FACTOR_DEFINITIONS = {
    "momentum_20d": "过去20个交易日收益率。",
    "momentum_60d": "过去60个交易日收益率。",
    "reversal_5d": "过去5个交易日收益率的相反数。",
    "low_volatility_20d": "20日收益率滚动标准差的相反数。",
    "low_turnover_20d": "20日平均换手率的相反数。",
    "low_liquidity_20d": "20日平均成交额取对数后的相反数。",
}

FACTOR_COLUMNS = list(FACTOR_DEFINITIONS)

FACTOR_LABELS = {
    "momentum_20d": "20日动量",
    "momentum_60d": "60日动量",
    "reversal_5d": "5日反转",
    "low_volatility_20d": "20日低波动",
    "low_turnover_20d": "20日低换手",
    "low_liquidity_20d": "20日低流动性",
}


def compute_features(panel: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    data = panel.sort_values(["code", "date"]).copy()
    grouped = data.groupby("code", sort=False)

    close = grouped["close"].ffill()
    data["momentum_20d"] = close.groupby(data["code"], sort=False).pct_change(
        20, fill_method=None
    )
    data["momentum_60d"] = close.groupby(data["code"], sort=False).pct_change(
        60, fill_method=None
    )
    past_5d = close.groupby(data["code"], sort=False).pct_change(
        5, fill_method=None
    )
    data["reversal_5d"] = -past_5d

    data["volatility_20d"] = grouped["ret"].transform(
        lambda values: values.rolling(20, min_periods=15).std()
    )
    data["low_volatility_20d"] = -data["volatility_20d"]

    data["turnover_20d"] = grouped["turnover"].transform(
        lambda values: values.rolling(20, min_periods=15).mean()
    )
    data["low_turnover_20d"] = -data["turnover_20d"]

    data["amount_20d"] = grouped["amount"].transform(
        lambda values: values.rolling(20, min_periods=15).mean()
    )
    data["low_liquidity_20d"] = -np.log1p(data["amount_20d"].clip(lower=0))

    for factor in FACTOR_COLUMNS:
        data[factor] = data[factor].replace([np.inf, -np.inf], np.nan)
        data[f"z_{factor}"] = _cross_sectional_winsor_zscore(data, factor)

    z_columns = [f"z_{factor}" for factor in FACTOR_COLUMNS]
    min_history = int(config["data"]["min_history_days"])
    min_adv = float(config["data"]["min_adv_20d"])
    data["history_count"] = grouped["valuation_close"].transform(
        lambda values: values.notna().cumsum()
    )
    data["eligible"] = (
        data["universe_member"].eq(1)
        & data["tradeable"]
        & data["history_count"].ge(min_history)
        & data["amount_20d"].ge(min_adv)
        & data[z_columns].notna().all(axis=1)
    )
    data["factor_score"] = data[z_columns].mean(axis=1, skipna=False)
    return data


def _cross_sectional_winsor_zscore(data: pd.DataFrame, column: str) -> pd.Series:
    def transform(values: pd.Series) -> pd.Series:
        valid = values.dropna()
        if len(valid) < 3:
            return pd.Series(np.nan, index=values.index, dtype=float)
        lower = valid.quantile(0.01)
        upper = valid.quantile(0.99)
        clipped = values.clip(lower, upper)
        std = clipped.std()
        if not np.isfinite(std) or std <= 1e-12:
            return pd.Series(0.0, index=values.index, dtype=float)
        return (clipped - clipped.mean()) / std

    return data.groupby("date", sort=False)[column].transform(transform)

