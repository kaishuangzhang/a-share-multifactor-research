from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def add_forward_labels(panel: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """
    A signal observed after close on T trades at open on T+1.

    The forward return therefore starts at T+1 open and exits after H trading days:
      open[T+1+H] / open[T+1] - 1
    """
    data = panel.sort_values(["code", "date"]).copy()
    horizon = int(config["labels"]["horizon_days"])
    grouped = data.groupby("code", sort=False)

    valid_open = data["open"].where(data["open"].gt(0))
    data["entry_open"] = valid_open.groupby(data["code"], sort=False).shift(-1)
    data["exit_open"] = valid_open.groupby(data["code"], sort=False).shift(
        -(horizon + 1)
    )
    data["future_return"] = data["exit_open"] / data["entry_open"] - 1.0
    data.loc[~np.isfinite(data["future_return"]), "future_return"] = np.nan
    data["label_end_date"] = grouped["date"].shift(-(horizon + 1))

    data["target_excess_return"] = (
        data["future_return"]
        - data.groupby("date", sort=False)["future_return"].transform("mean")
    )
    data["target_rank"] = (
        data.groupby("date", sort=False)["target_excess_return"].rank(
            method="average", pct=True
        )
        - 0.5
    )
    return data

