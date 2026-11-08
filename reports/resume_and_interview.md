# Resume and interview notes

These notes use only the reproducible artifacts in this repository. Keep the
data limitation visible whenever quoting a performance number.

## Recommended resume bullets

- Built a reproducible A-share cross-sectional research pipeline over 287
  current sample constituents and 417,872 aligned stock-day records, including
  data validation, six technical factors, cross-sectional normalization, and
  monthly Top-20 portfolio construction.
- Designed close-to-open signal timing with T+1 execution, a 20-trading-day
  forward target, purged train-validation-test splits, and automated leakage
  tests to prevent training labels from crossing validation or test periods.
- Implemented a unified backtest with board-lot rounding, commissions,
  date-dependent stamp tax, slippage, cash constraints, approximate price
  limits, and a daily traded-amount cap; compared a factor baseline and a
  gradient-boosting model against an equal-weight universe benchmark.
- On the 2023-2024 holdout, the six-factor baseline had an annualized return of
  31.3%, maximum drawdown of -10.8%, and 24.4% cumulative active return versus
  the equal-weight sample universe after modeled costs.
- Diagnosed that better validation Rank IC for the gradient-boosting model
  (0.046 versus 0.015) did not translate into stronger holdout portfolio
  returns, demonstrating the importance of evaluating portfolio-level,
  cost-aware evidence rather than model metrics alone.

## What not to claim

- Do not call the result "CSI 300 alpha"; the benchmark is not the CSI 300
  index and the universe is not historical index membership.
- Do not call the strategy production ready; the data still contains
  survivorship bias and lacks industry, ST, and exact exchange limit flags.
- Do not say the machine-learning model beat the factor baseline. In the final
  holdout, the baseline had the higher return and lower drawdown.
- Do not reuse the old full-sample annualized return. This project's final
  evidence is the 2023-2024 holdout.

## Key interview talking points

### Why T+1 execution matters

The signal uses the T close, which is available only after the close. Trading
at that same close would use information that could not have supported the
order. The label therefore starts at the next open.

### Why labels are purged

A 20-day forward return on a late-December date includes January prices. The
split logic removes any training or validation row whose `label_end_date`
crosses into the next period.

### Why the simple baseline matters

Machine-learning validation Rank IC was higher, but portfolio returns were
lower. A credible research process reports both results. The model may be
overfitting factor interactions that do not survive portfolio construction and
costs.

### Why turnover is high

The portfolio rebalances monthly and can replace many Top-20 names. One-way
annual turnover is above seven times, making transaction costs and capacity
central rather than secondary considerations.

## Highest-value next improvements

1. Replace the current constituent file list with point-in-time index
   membership and include delisted stocks.
2. Add the actual CSI 300 index as the primary benchmark once its history is
   available.
3. Add industry and size neutralization.
4. Run expanding-window walk-forward testing and a transaction-cost sensitivity
   grid.
5. Add a proper risk model and stability analysis across subperiods.
