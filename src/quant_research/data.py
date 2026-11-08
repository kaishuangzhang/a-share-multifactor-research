from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import resolve_project_path


REQUIRED_RAW_COLUMNS = [
    "date",
    "open",
    "close",
    "high",
    "low",
    "volume",
    "amount",
    "turnover",
]


@dataclass(frozen=True)
class DataQualityReport:
    stock_files: int
    rows_loaded: int
    rows_aligned: int
    invalid_ohlc_rows: int
    non_positive_close_rows: int
    duplicate_stock_date_rows: int
    coverage_ratio: float


def load_and_clean_market_data(
    config: dict[str, Any],
    output_dir: Path,
) -> tuple[pd.DataFrame, DataQualityReport]:
    """Load raw files, align every stock to one trading calendar, and save a clean panel."""
    raw_dir = resolve_project_path(config["data"]["raw_dir"])
    files = sorted(raw_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {raw_dir}")

    start_date = pd.Timestamp(config["data"]["start_date"])
    end_date = pd.Timestamp(config["data"]["end_date"])
    frames: list[pd.DataFrame] = []
    quality = {
        "stock_files": len(files),
        "rows_loaded": 0,
        "invalid_ohlc_rows": 0,
        "non_positive_close_rows": 0,
        "duplicate_stock_date_rows": 0,
    }

    for path in files:
        code = path.stem.zfill(6)
        raw = pd.read_parquet(path)
        missing_columns = set(REQUIRED_RAW_COLUMNS).difference(raw.columns)
        if missing_columns:
            raise ValueError(f"{path.name} missing columns: {sorted(missing_columns)}")

        frame = raw[REQUIRED_RAW_COLUMNS].copy()
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        for column in REQUIRED_RAW_COLUMNS[1:]:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

        quality["rows_loaded"] += len(frame)
        quality["duplicate_stock_date_rows"] += int(frame.duplicated("date").sum())
        quality["invalid_ohlc_rows"] += int(
            (
                (frame["high"] < frame["low"])
                | (frame["high"] < frame["open"])
                | (frame["high"] < frame["close"])
                | (frame["low"] > frame["open"])
                | (frame["low"] > frame["close"])
            ).sum()
        )
        quality["non_positive_close_rows"] += int((frame["close"] <= 0).sum())

        frame = frame.dropna(subset=["date"]).drop_duplicates("date", keep="last")
        frame = frame[(frame["date"] >= start_date) & (frame["date"] <= end_date)]
        frame["code"] = code
        frames.append(frame.sort_values("date"))

    combined = pd.concat(frames, ignore_index=True)
    calendar = pd.DatetimeIndex(sorted(combined["date"].unique()), name="date")

    aligned_frames: list[pd.DataFrame] = []
    for code, frame in combined.groupby("code", sort=True):
        frame = frame.set_index("date").reindex(calendar)
        frame["code"] = code
        aligned_frames.append(frame.reset_index())

    panel = pd.concat(aligned_frames, ignore_index=True)
    panel = panel.sort_values(["code", "date"]).reset_index(drop=True)
    panel["universe_member"] = 1
    panel["data_valid"] = (
        panel[["open", "close", "high", "low"]].gt(0).all(axis=1)
        & panel[["volume", "amount", "turnover"]].ge(0).all(axis=1)
        & panel["high"].ge(panel["low"])
    )

    grouped = panel.groupby("code", sort=False)
    positive_close = panel["close"].where(panel["close"].gt(0))
    panel["valuation_close"] = positive_close.groupby(panel["code"], sort=False).transform(
        lambda values: values.ffill()
    )
    panel["prev_close"] = panel.groupby("code", sort=False)["valuation_close"].shift(1)
    panel["ret"] = panel.groupby("code", sort=False)["valuation_close"].pct_change(
        fill_method=None
    )
    panel.loc[~np.isfinite(panel["ret"]), "ret"] = np.nan
    panel["tradeable"] = (
        panel["data_valid"]
        & panel["open"].gt(0)
        & panel["amount"].gt(0)
        & panel["volume"].gt(0)
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(output_dir / "market_panel.parquet", index=False)

    report = DataQualityReport(
        stock_files=len(files),
        rows_loaded=int(quality["rows_loaded"]),
        rows_aligned=len(panel),
        invalid_ohlc_rows=int(quality["invalid_ohlc_rows"]),
        non_positive_close_rows=int(quality["non_positive_close_rows"]),
        duplicate_stock_date_rows=int(quality["duplicate_stock_date_rows"]),
        coverage_ratio=float(panel["data_valid"].mean() if len(panel) else 0.0),
    )
    with (output_dir / "data_quality.json").open("w", encoding="utf-8") as handle:
        json.dump(report.__dict__, handle, ensure_ascii=False, indent=2)

    return panel, report


def build_equal_weight_benchmark(panel: pd.DataFrame) -> pd.DataFrame:
    """Build a transparent equal-weight universe benchmark from close-to-close returns."""
    eligible = panel.loc[
        panel["universe_member"].eq(1)
        & panel["data_valid"]
        & panel["ret"].notna(),
        ["date", "code", "ret"],
    ]
    benchmark_ret = eligible.groupby("date")["ret"].mean().rename("benchmark_ret")
    benchmark = benchmark_ret.reset_index()
    benchmark["benchmark_nav"] = (1.0 + benchmark["benchmark_ret"]).cumprod()
    return benchmark
