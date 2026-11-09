from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from .features import FACTOR_LABELS


def create_research_charts(
    equity_curves: dict[str, pd.DataFrame],
    benchmark: pd.DataFrame,
    factor_summary: pd.DataFrame,
    bucket_returns: pd.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    _plot_equity_and_drawdown(equity_curves, benchmark, output_dir)
    _plot_factor_ic(factor_summary, output_dir)
    _plot_factor_buckets(bucket_returns, output_dir)


def _plot_equity_and_drawdown(
    equity_curves: dict[str, pd.DataFrame],
    benchmark: pd.DataFrame,
    output_dir: Path,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    benchmark_curve: pd.DataFrame | None = None
    for strategy, equity in equity_curves.items():
        merged = equity.merge(benchmark[["date", "benchmark_ret"]], on="date", how="left")
        merged["benchmark_ret"] = merged["benchmark_ret"].fillna(0.0)
        merged["benchmark_nav_rebased"] = (1.0 + merged["benchmark_ret"]).cumprod()
        if benchmark_curve is None:
            benchmark_curve = merged[["date", "benchmark_nav_rebased"]].copy()
        strategy_label = "简单因子基线" if strategy == "baseline" else "梯度提升模型"
        axes[0].plot(
            merged["date"], merged["nav"], label=strategy_label, linewidth=1.4
        )
        axes[1].plot(
            merged["date"],
            merged["nav"] / merged["benchmark_nav_rebased"] - 1.0,
            label=f"{strategy_label}主动收益",
            linewidth=1.2,
        )
    if benchmark_curve is not None:
        axes[0].plot(
            benchmark_curve["date"],
            benchmark_curve["benchmark_nav_rebased"],
            label="等权股票池",
            color="black",
            linestyle="--",
            linewidth=1.2,
        )
    axes[0].set_title("样本外净值曲线")
    axes[0].set_ylabel("净值")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_title("相对等权股票池的主动收益")
    axes[1].set_ylabel("主动收益")
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "equity_and_active_return.png", dpi=160)
    plt.close(fig)


def _plot_factor_ic(factor_summary: pd.DataFrame, output_dir: Path) -> None:
    if factor_summary.empty:
        return
    data = factor_summary.sort_values("mean_rank_ic").copy()
    data["factor"] = data["factor"].map(FACTOR_LABELS)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(data["factor"], data["mean_rank_ic"], color="#2878b5")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("训练期与验证期平均日排名信息系数")
    ax.set_xlabel("平均排名信息系数")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "factor_rank_ic.png", dpi=160)
    plt.close(fig)


def _plot_factor_buckets(bucket_returns: pd.DataFrame, output_dir: Path) -> None:
    if bucket_returns.empty:
        return
    chart_data = bucket_returns.copy()
    chart_data["factor"] = chart_data["factor"].map(FACTOR_LABELS)
    pivot = chart_data.pivot(
        index="factor", columns="bucket", values="future_excess_return"
    )
    fig, ax = plt.subplots(figsize=(11, 5))
    for factor, row in pivot.iterrows():
        ax.plot(row.index, row.values, marker="o", label=factor)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("各因子五分组平均未来超额收益")
    ax.set_xlabel("因子分组（越往右因子分数越高）")
    ax.set_ylabel("平均未来超额收益")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "factor_quintile_returns.png", dpi=160)
    plt.close(fig)

