# A-share cross-sectional factor research

This project is a reproducible daily-data research pipeline for A-share
cross-sectional stock selection. It is intentionally built around one clean
research design rather than many incompatible backtests.

## Research design

- A signal is formed after the close on day `T`.
- Orders execute at the opening price on the next trading day, `T+1`.
- The prediction target is the 20-trading-day return from the `T+1` open.
- Stocks are ranked cross-sectionally within the same trading day.
- Portfolios hold the Top 20 names with equal weights and monthly rebalancing.
- Training uses 2019-2021, validation uses 2022, and the final holdout is 2023-2024.
- Training and validation labels are purged before the next period starts.
- The backtest includes board-lot rounding, commissions, date-dependent stamp
  tax, slippage, cash constraints, price-limit checks, and a traded-amount cap.

The forward label is:

```text
future_return(T) = open(T+1+20) / open(T+1) - 1
```

It deliberately excludes the return between the T close and the T+1 open
because that return cannot be captured by a close-based signal.

## Factors

All factors are oriented so that a larger value means a more favorable prior:

| Factor | Definition | Intuition |
|---|---|---|
| `momentum_20d` | Past 20-day return | Medium-term price trend |
| `momentum_60d` | Past 60-day return | Longer trend |
| `reversal_5d` | Negative past 5-day return | Short-term oversold reversal |
| `low_volatility_20d` | Negative 20-day return volatility | Low-volatility effect |
| `low_turnover_20d` | Negative 20-day average turnover | Crowding and speculation |
| `low_liquidity_20d` | Negative log average traded amount | Illiquidity premium |

Each raw factor is winsorized and standardized within the same-date
cross-section. The simple baseline is the equal-weight average of these
standardized factors. The machine-learning challenger is a scikit-learn
histogram gradient-boosting regressor trained to predict cross-sectional
relative-return ranks.

## Quick start

From this directory:

```powershell
python -m pip install -r requirements.txt
python -m pytest -q
python scripts/run_pipeline.py
```

The pipeline reads raw parquet files from `../data/raw`. It does not use the
old processed dataset.

For a chapter-by-chapter beginner explanation of the data, factors, labels,
model, portfolio, backtest, metrics, and limitations, read
[docs/beginner_project_walkthrough.md](docs/beginner_project_walkthrough.md).

## Outputs

- `reports/research_report.md`: concise research report and results.
- `reports/equity_and_active_return.png`: test-period equity and active return.
- `reports/factor_rank_ic.png`: factor Rank IC summary.
- `reports/factor_quintile_returns.png`: factor bucket analysis.
- `outputs/performance_summary.csv`: strategy metrics.
- `outputs/factor_rank_ic_summary.csv`: factor evidence.
- `outputs/trade_log.parquet`: every simulated order.
- `outputs/rebalance_targets.parquet`: monthly signal and trade dates.
- `outputs/research_panel.parquet`: features and forward labels.

## Tests

The tests cover the highest-risk timing assumptions:

- labels begin at the next open and end after the stated horizon;
- training and validation labels are purged at split boundaries;
- portfolio trade dates are strictly after signal dates;
- all target weights sum to the configured gross exposure;
- the backtest creates no trades before the first target trade date.

## Data limitations

The supplied raw dataset contains 287 current constituent files. It does not
contain historical index membership or delisted stocks, so survivorship bias
remains. The included benchmark is an equal-weight benchmark of the available
universe, not a tradable CSI 300 index. Industry exposure, ST status, and exact
exchange limit flags are also unavailable.

These limitations must stay visible when discussing results. A positive
backtest is not evidence of a production-ready strategy without point-in-time
membership and a tradable index benchmark.

## Project layout

```text
configs/                 research parameters
src/quant_research/      data, factors, labels, model, portfolio, backtest
scripts/run_pipeline.py  one-command pipeline
tests/                   leakage and execution tests
outputs/                 reproducible data and numerical outputs
reports/                 report, charts, and presentation material
```

