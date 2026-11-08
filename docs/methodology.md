# Beginner methodology guide

## 1. What the project predicts

The model predicts which stocks are likely to outperform the stock universe
over the next month. It does not predict whether the overall market will rise.

At each date and for each stock, the pipeline builds:

```text
features observed through T close -> rank of future excess return after T+1
```

The cross-sectional target subtracts the same-date average future return. This
makes the model focus on relative selection rather than market direction.

## 2. Why `T` and `T+1` matter

Closing prices are only known after the market closes. A signal using the T
close cannot buy at that same close. The first realistic execution point is
the next trading day's open.

The supplied backtest therefore requires every order to satisfy:

```text
signal_date < trade_date
```

A backtest that earns the return from T close to T+1 open from a T-close signal
has look-ahead bias.

## 3. Why the data is split by date

Randomly splitting rows would place neighboring observations from the same
stock and time window in both training and test data. A 20-day forward label
also overlaps across neighboring dates.

This project:

1. trains on 2019-2021;
2. removes training labels that extend into 2022;
3. validates on 2022;
4. removes validation labels that extend into 2023;
5. treats 2023-2024 as the final test period.

The `label_end_date` test is the practical guard against horizon leakage.

## 4. What Rank IC means

For each date:

```text
Rank IC = Spearman correlation(factor rank, future-return rank)
```

Positive Rank IC means higher factor values tended to be followed by higher
future returns. A stable positive IC is more useful than a high IC on one or
two dates.

The pipeline reports the mean, standard deviation, ICIR, positive ratio, and
the number of valid days. It also plots average future excess returns by
factor quintile.

## 5. Why a simple baseline is mandatory

A complex model can look impressive because of leakage, transaction costs, or
market exposure. The project therefore compares three layers:

1. the available stock universe;
2. an equal-weight multi-factor score;
3. a gradient-boosting challenger.

The model is useful only if it adds evidence beyond the simple factor
baseline. In the current run, the simple baseline is stronger out of sample;
that is a valid research result, not something to hide.

## 6. What the backtest includes

- the portfolio is formed monthly;
- orders use the next trading day's open;
- share counts round to A-share board lots;
- commissions have a minimum charge;
- the sell-side stamp tax changes on 2023-08-28;
- slippage is applied to both buys and sells;
- purchases cannot exceed available cash;
- trades cannot consume more than 5% of daily traded amount;
- approximate exchange price limits block unavailable fills;
- missing or suspended prices use the last available valuation price.

## 7. How to read the final report

Start with the validation Rank IC, then the out-of-sample return and drawdown,
and only then compare the return with the benchmark. Always inspect:

- active return, not just absolute return;
- turnover and explicit costs;
- different calendar years;
- the simple baseline;
- the survivorship-bias limitation.

The correct conclusion may be that the model does not beat the baseline. That
is still a stronger interview project than an unexplained high return.
