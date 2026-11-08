from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    _validate_config(config)
    return config


def resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (PROJECT_ROOT / candidate).resolve()


def _validate_config(config: dict[str, Any]) -> None:
    required_sections = {"data", "labels", "splits", "portfolio", "backtest"}
    missing = required_sections.difference(config)
    if missing:
        raise ValueError(f"Missing config sections: {sorted(missing)}")

    label_horizon = int(config["labels"]["horizon_days"])
    if label_horizon != int(config["labels"]["horizon_days"]) or label_horizon <= 0:
        raise ValueError("labels.horizon_days must be a positive integer")

    if float(config["portfolio"]["gross_exposure"]) <= 0:
        raise ValueError("portfolio.gross_exposure must be positive")

    top_n = int(config["portfolio"]["top_n"])
    min_stocks = int(config["portfolio"]["min_stocks"])
    if top_n < min_stocks:
        raise ValueError("portfolio.top_n must be at least portfolio.min_stocks")

