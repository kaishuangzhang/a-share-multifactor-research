from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .backtest import run_backtests
from .config import load_config, resolve_project_path
from .data import build_equal_weight_benchmark, load_and_clean_market_data
from .factor_analysis import run_factor_analysis
from .features import FACTOR_DEFINITIONS, FACTOR_LABELS, compute_features
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

    print("[1/7] 正在加载并校验原始行情数据")
    panel, quality = load_and_clean_market_data(config, output_dir)

    print("[2/7] 正在计算因子和未来收益标签")
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
        [
            {
                "因子代码": factor,
                "中文名称": FACTOR_LABELS[factor],
                "定义": definition,
            }
            for factor, definition in FACTOR_DEFINITIONS.items()
        ]
    ).to_csv(output_dir / "factor_dictionary.csv", index=False)

    print("[3/7] 正在创建剔除跨期标签的时间序列切分")
    train_mask, validation_mask, test_mask, split_stats = make_split_masks(
        research, config
    )
    research_mask = train_mask | validation_mask
    with (output_dir / "split_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(split_stats.__dict__, handle, ensure_ascii=False, indent=2, default=str)

    print("[4/7] 正在运行因子排名信息系数和五分组分析")
    _, factor_summary, bucket_returns, factor_correlation = run_factor_analysis(
        research, research_mask, output_dir
    )

    print("[5/7] 正在拟合简单基线和梯度提升模型")
    scored, validation_summary = fit_baseline_and_model(
        research, train_mask, validation_mask, config, output_dir
    )

    print("[6/7] 正在构建月度组合并运行T+1回测")
    targets = build_monthly_targets(scored, config, output_dir)
    equity_curves, trades = run_backtests(scored, targets, config, output_dir)

    print("[7/7] 正在计算绩效并生成报告")
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
    factor_table = factor_summary.rename(
        columns={
            "factor": "因子",
            "mean_rank_ic": "平均排名信息系数",
            "std_rank_ic": "排名信息系数标准差",
            "icir": "信息系数稳定度",
            "positive_ic_ratio": "正系数比例",
            "median_observations": "每日样本中位数",
            "days": "有效交易日",
        }
    )
    factor_table["因子"] = factor_table["因子"].map(FACTOR_LABELS)
    correlation_table = factor_correlation.rename(
        columns={
            "factor_left": "因子A",
            "factor_right": "因子B",
            "correlation": "平均相关系数",
        }
    )
    correlation_table["因子A"] = correlation_table["因子A"].map(FACTOR_LABELS)
    correlation_table["因子B"] = correlation_table["因子B"].map(FACTOR_LABELS)
    performance_table = performance.rename(
        columns={
            "strategy": "策略",
            "initial_cash": "初始资金",
            "final_value": "期末资产",
            "total_return": "总收益",
            "annual_return": "年化收益",
            "annual_volatility": "年化波动",
            "max_drawdown": "最大回撤",
            "sharpe": "夏普比率",
            "calmar": "卡玛比率",
            "daily_win_rate": "日胜率",
            "benchmark_return": "基准收益",
            "active_return": "主动收益",
            "tracking_error": "跟踪误差",
            "information_ratio": "信息比率",
            "annual_one_way_turnover": "年化单边换手率",
            "total_explicit_cost": "总显性及滑点成本",
            "days": "交易日数",
        }
    )
    performance_table["策略"] = performance_table["策略"].map(
        {"baseline": "简单因子基线", "model": "梯度提升模型"}
    )
    yearly_table = yearly.rename(
        columns={
            "strategy": "策略",
            "year": "年份",
            "return": "收益",
            "max_drawdown": "最大回撤",
        }
    )
    yearly_table["策略"] = yearly_table["策略"].map(
        {"baseline": "简单因子基线", "model": "梯度提升模型"}
    )
    model_validation = validation_summary["model"]
    baseline_validation = validation_summary["baseline"]

    lines = [
        "# A股横截面因子研究",
        "",
        "## 研究设计",
        "",
        "- 在T日收盘后生成信号，在T+1日开盘执行交易。",
        "- 未来收益标签从T+1日开盘开始，覆盖20个交易日。",
        "- 模型使用2019至2021年数据训练，使用2022年验证，使用2023至2024年进行最终样本外测试。",
        "- 训练集和验证集会剔除未来标签跨越下一时期边界的样本。",
        "- 组合按月调仓，持有得分最高的20只股票，并使用等权配置。",
        "",
        "## 数据边界",
        "",
        f"- 股票文件数量：{quality.stock_files}",
        f"- 统一交易日历后的面板行数：{quality.rows_aligned}",
        f"- 有效行情覆盖率：{quality.coverage_ratio:.2%}",
        "- 当前原始股票池由当前成分文件构成，因此仍存在幸存者偏差。",
        "- 当前基准为可用股票池的等权收益，不是可交易的沪深300指数。",
        "",
        "## 数据切分",
        "",
        f"- 训练集行数：{split_stats.train_rows}",
        f"- 验证集行数：{split_stats.validation_rows}",
        f"- 训练标签最晚结束日期：{split_stats.train_label_end_max}",
        f"- 验证标签最晚结束日期：{split_stats.validation_label_end_max}",
        "",
        "## 因子证据",
        "",
        factor_table.to_markdown(index=False),
        "",
        "## 相关性最强的因子组合",
        "",
        correlation_table.reindex(
            factor_correlation["correlation"].abs().sort_values(ascending=False).index
        )
        .head(10)
        .to_markdown(index=False),
        "",
        "## 验证期排名信息系数",
        "",
        (
            "- 梯度提升模型："
            f"平均排名信息系数={model_validation['mean_rank_ic']:.4f}，"
            f"信息系数稳定度={model_validation['icir']:.3f}，"
            f"正系数比例={model_validation['positive_ratio']:.2%}，"
            f"有效交易日={model_validation['days']}"
        ),
        (
            "- 等权因子基线："
            f"平均排名信息系数={baseline_validation['mean_rank_ic']:.4f}，"
            f"信息系数稳定度={baseline_validation['icir']:.3f}，"
            f"正系数比例={baseline_validation['positive_ratio']:.2%}，"
            f"有效交易日={baseline_validation['days']}"
        ),
        "",
        "## 样本外绩效",
        "",
        performance_table.to_markdown(index=False),
        "",
        "## 分年度绩效",
        "",
        yearly_table.to_markdown(index=False),
        "",
        "## 研究限制",
        "",
        "- 当前数据不包含历史时点指数成分信息。",
        "- 当前数据不包含行业分类、ST状态和交易所精确涨跌停标志。",
        "- 等权股票池基准不是可交易指数。",
        "- 无论是正收益还是负收益，样本外结果都应如实保留，不能用样本内结果替换。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

