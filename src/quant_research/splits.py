from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class SplitStats:
    train_rows: int
    validation_rows: int
    test_rows: int
    train_label_end_max: pd.Timestamp
    validation_label_end_max: pd.Timestamp


def make_split_masks(
    data: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.Series, pd.Series, pd.Series, SplitStats]:
    split = config["splits"]
    dates = data["date"]
    train_start = pd.Timestamp(split["train_start"])
    train_end = pd.Timestamp(split["train_end"])
    validation_start = pd.Timestamp(split["validation_start"])
    validation_end = pd.Timestamp(split["validation_end"])
    test_start = pd.Timestamp(split["test_start"])
    test_end = pd.Timestamp(split["test_end"])

    # A training label may not look into the validation period.
    train_mask = (
        dates.between(train_start, train_end)
        & data["label_end_date"].notna()
        & (data["label_end_date"] < validation_start)
    )
    # A validation label may not look into the final test period.
    validation_mask = (
        dates.between(validation_start, validation_end)
        & data["label_end_date"].notna()
        & (data["label_end_date"] < test_start)
    )
    test_mask = dates.between(test_start, test_end)

    train_end_max = data.loc[train_mask, "label_end_date"].max()
    validation_end_max = data.loc[validation_mask, "label_end_date"].max()
    stats = SplitStats(
        train_rows=int(train_mask.sum()),
        validation_rows=int(validation_mask.sum()),
        test_rows=int(test_mask.sum()),
        train_label_end_max=train_end_max,
        validation_label_end_max=validation_end_max,
    )
    return train_mask, validation_mask, test_mask, stats

