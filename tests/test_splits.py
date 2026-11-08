from __future__ import annotations

import pandas as pd

from quant_research.splits import make_split_masks


def test_training_and_validation_labels_are_purged() -> None:
    dates = pd.bdate_range("2021-01-01", "2023-12-31")
    data = pd.DataFrame(
        {
            "date": dates,
            "label_end_date": dates,
        }
    )
    config = {
        "splits": {
            "train_start": "2021-01-01",
            "train_end": "2021-12-31",
            "validation_start": "2022-01-01",
            "validation_end": "2022-12-31",
            "test_start": "2023-01-01",
            "test_end": "2023-12-31",
        }
    }
    train, validation, test, _ = make_split_masks(data, config)
    assert data.loc[train, "label_end_date"].max() < pd.Timestamp("2022-01-01")
    assert data.loc[validation, "label_end_date"].max() < pd.Timestamp("2023-01-01")
    assert data.loc[test, "date"].min() >= pd.Timestamp("2023-01-01")

