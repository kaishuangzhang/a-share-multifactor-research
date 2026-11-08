from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .backtest import run_backtests
from .config import load_config, resolve_project_path
from .data import build_equal_weight_benchmark, load_and_clean_market_data
from .factor_analysis import run_factor_analysis
from .features import FACTOR_DEFINITIONS, compute_features
from .labels import add_forward_labels
from .metrics import evaluate_strategies
from .models import fit_baseline_and_model
from .portfolio import build_monthly_targets
from .splits import make_split_masks
from .visualization import create_research_charts


def run_pipeline(config_path: str | Path = "configs/base.json") -> dict[str, Any]:
    config = load_config(config_path)
    output_dir = resolve_project_path("outputs")
    report_dir = resolve_project_path("reports")
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    print("[1/7] Loading and validating raw market data")
    panel, quality = load_and_clean_market_data(config, output_dir)

    print("[2/7] Computing factors and forward labels")
    featured = compute_features(panel, config)
    research = add_forward_labels(featured, config)
    panel_columns = [
        "date",
        "code",
        "open",
        "close",
        "high",
        "low",
        "volume",
        "amount",
        "turnover",
        "prev_close",
        "valuation_close",
        "ret",
        "tradeable",
        "history_count",
        "amount_20d",
        "volatility_20d",
        "turnover_20d",
        "momentum_20d",
        "momentum_60d",
        "reversal_5d",
        "low_volatility_20d",
        "low_turnover_20d",
        "low_liquidity_20d",
        "eligible",
        "factor_score",
        "entry_open",
        "exit_open",
        "future_return",
        "target_excess_return",
        "target_rank",
        "label_end_date",
        *[f"z_{factor}" for factor in FACTOR_DEFINITIONS],
    ]
    research[panel_columns].to_parquet(
        output_dir / "research_panel.parquet", index=False
    )
    pd.DataFrame(
        [{"factor": factor, "definition": definition} for factor, definition in FACTOR_DEFINITIONS.items()]
    ).to_csv(output_dir / "factor_dictionary.csv", index=False)

    print("[3/7] Creating purged time-series splits")
    train_mask, validation_mask, test_mask, split_stats = make_split_masks(
        research, config
    )
    research_mask = train_mask | validation_mask
    with (output_dir / "split_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(split_stats.__dict__, handle, ensure_ascii=False, indent=2, default=str)

    print("[4/7] Running factor Rank IC and quintile analysis")
    _, factor_summary, bucket_returns, factor_correlation = run_factor_analysis(
        research, research_mask, output_dir
    )

    print("[5/7] Fitting baseline and gradient-boosting model")
    scored, validation_summary = fit_baseline_and_model(
        research, train_mask, validation_mask, config, output_dir
    )

    print("[6/7] Building monthly portfolios and running T+1 backtests")
    targets = build_monthly_targets(scored, config, output_dir)
    equity_curves, trades = run_backtests(scored, targets, config, output_dir)

    print("[7/7] Evaluating and writing reports")
    benchmark = build_equal_weight_benchmark(scored)
    benchmark.to_parquet(output_dir / "benchmark.parquet", index=False)
    performance, yearly = evaluate_strategies(
        equity_curves, benchmark, trades, config, output_dir
    )
    create_research_charts(
        equity_curves,
        benchmark,
        factor_summary,
        bucket_returns,
        report_dir,
    )
    write_research_report(
        report_dir / "research_report.md",
        quality=quality,
        split_stats=split_stats,
        validation_summary=validation_summary,
        factor_summary=factor_summary,
        factor_correlation=factor_correlation,
        performance=performance,
        yearly=yearly,
    )

    run_summary = {
        "data_quality": quality.__dict__,
        "splits": split_stats.__dict__,
        "validation": validation_summary,
        "performance": performance.to_dict(orient="records"),
        "outputs": str(output_dir),
        "reports": str(report_dir),
    }
    with (output_dir / "run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(run_summary, handle, ensure_ascii=False, indent=2, default=str)
    return run_summary


def write_research_report(
    path: Path,
    *,
    quality: Any,
    split_stats: Any,
    validation_summary: dict[str, Any],
    factor_summary: pd.DataFrame,
    factor_correlation: pd.DataFrame,
    performance: pd.DataFrame,
    yearly: pd.DataFrame,
) -> None:
    lines = [
        "# A-share cross-sectional factor research",
        "",
        "## Research design",
        "",
        "- Signals are formed after the close on T and trades execute at the open on T+1.",
        "- The forward target spans 20 trading days from the T+1 open.",
        "- Model training uses 2019-2021, validation uses 2022, and the final test uses 2023-2024.",
        "- Training and validation samples are purged when their forward label crosses the next period.",
        "- Portfolios rebalance monthly, hold Top 20 names, and use equal weights.",
        "",
        "## Data boundary",
        "",
        f"- Stock files: {quality.stock_files}",
        f"- Aligned panel rows: {quality.rows_aligned}",
        f"- Trading-calendar coverage: {quality.coverage_ratio:.2%}",
        "- The supplied raw universe contains current constituent files, so survivorship bias remains.",
        "- The included benchmark is an equal-weight benchmark of the available universe.",
        "",
        "## Data splits",
        "",
        f"- Training rows: {split_stats.train_rows}",
        f"- Validation rows: {split_stats.validation_rows}",
        f"- Training label end maximum: {split_stats.train_label_end_max}",
        f"- Validation label end maximum: {split_stats.validation_label_end_max}",
        "",
        "## Factor evidence",
        "",
        factor_summary.to_markdown(index=False),
        "",
        "## Largest absolute factor correlations",
        "",
        factor_correlation.reindex(
            factor_correlation["correlation"].abs().sort_values(ascending=False).index
        )
        .head(10)
        .to_markdown(index=False),
        "",
        "## Validation Rank IC",
        "",
        f"- Model: {validation_summary['model']}",
        f"- Equal-weight factor baseline: {validation_summary['baseline']}",
        "",
        "## Out-of-sample performance",
        "",
        performance.to_markdown(index=False),
        "",
        "## Yearly performance",
        "",
        yearly.to_markdown(index=False),
        "",
        "## Limitations",
        "",
        "- Historical index membership is not present in the supplied data.",
        "- Industry classification, ST status, and exact exchange price-limit flags are not present.",
        "- The equal-weight universe benchmark is not a tradable index.",
        "- A negative out-of-sample result remains evidence; do not replace it with an in-sample result.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

