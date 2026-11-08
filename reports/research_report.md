# A-share cross-sectional factor research

## Research design

- Signals are formed after the close on T and trades execute at the open on T+1.
- The forward target spans 20 trading days from the T+1 open.
- Model training uses 2019-2021, validation uses 2022, and the final test uses 2023-2024.
- Training and validation samples are purged when their forward label crosses the next period.
- Portfolios rebalance monthly, hold Top 20 names, and use equal weights.

## Data boundary

- Stock files: 287
- Aligned panel rows: 417872
- Trading-calendar coverage: 92.99%
- The supplied raw universe contains current constituent files, so survivorship bias remains.
- The included benchmark is an equal-weight benchmark of the available universe.

## Data splits

- Training rows: 203483
- Validation rows: 63427
- Training label end maximum: 2021-12-31 00:00:00
- Validation label end maximum: 2022-12-30 00:00:00

## Factor evidence

| factor             |   mean_rank_ic |   std_rank_ic |        icir |   positive_ic_ratio |   median_observations |   days |
|:-------------------|---------------:|--------------:|------------:|--------------------:|----------------------:|-------:|
| low_liquidity_20d  |    0.0254581   |      0.176326 |  0.144381   |            0.54023  |                   257 |    870 |
| reversal_5d        |    0.00768991  |      0.168056 |  0.0457581  |            0.511494 |                   257 |    870 |
| low_turnover_20d   |   -0.000404887 |      0.245414 | -0.00164981 |            0.46092  |                   257 |    870 |
| momentum_60d       |   -0.0231241   |      0.200791 | -0.115165   |            0.488506 |                   257 |    870 |
| momentum_20d       |   -0.0286479   |      0.17783  | -0.161097   |            0.482759 |                   257 |    870 |
| low_volatility_20d |   -0.0474017   |      0.218471 | -0.21697    |            0.393103 |                   257 |    870 |

## Largest absolute factor correlations

| factor_left        | factor_right       |   correlation |
|:-------------------|:-------------------|--------------:|
| low_volatility_20d | low_turnover_20d   |      0.580549 |
| momentum_20d       | momentum_60d       |      0.496599 |
| momentum_20d       | reversal_5d        |     -0.440469 |
| low_turnover_20d   | low_liquidity_20d  |      0.380713 |
| low_volatility_20d | low_liquidity_20d  |      0.342545 |
| momentum_60d       | low_volatility_20d |     -0.341506 |
| momentum_20d       | low_volatility_20d |     -0.248841 |
| momentum_60d       | low_liquidity_20d  |     -0.237685 |
| momentum_60d       | reversal_5d        |     -0.232903 |
| momentum_60d       | low_turnover_20d   |     -0.178433 |

## Validation Rank IC

- Model: {'mean_rank_ic': 0.045525146246353135, 'std_rank_ic': 0.12941263779115045, 'icir': 0.3517828476676507, 'positive_ratio': 0.6244343891402715, 'days': 221}
- Equal-weight factor baseline: {'mean_rank_ic': 0.015033427020964451, 'std_rank_ic': 0.22508605755466965, 'icir': 0.06678968561752467, 'positive_ratio': 0.5384615384615384, 'days': 221}

## Out-of-sample performance

| strategy   |   initial_cash |   final_value |   total_return |   annual_return |   annual_volatility |   max_drawdown |   sharpe |   calmar |   daily_win_rate |   benchmark_return |   active_return |   tracking_error |   information_ratio |   annual_one_way_turnover |   total_explicit_cost |   days |
|:-----------|---------------:|--------------:|---------------:|----------------:|--------------------:|---------------:|---------:|---------:|-----------------:|-------------------:|----------------:|-----------------:|--------------------:|--------------------------:|----------------------:|-------:|
| baseline   |          1e+06 |   1.65904e+06 |       0.659035 |        0.31336  |            0.171913 |      -0.107738 | 1.6712   | 2.90854  |         0.525641 |            0.33406 |       0.243598  |         0.145762 |            0.755559 |                   7.17274 |               52612.8 |    468 |
| model      |          1e+06 |   1.4639e+06  |       0.4639   |        0.227783 |            0.283582 |      -0.244107 | 0.863623 | 0.933128 |         0.504274 |            0.33406 |       0.0973267 |         0.151189 |            0.448036 |                   7.85321 |               54104.2 |    468 |

## Yearly performance

| strategy   |   year |   return |   max_drawdown |
|:-----------|-------:|---------:|---------------:|
| baseline   |   2023 | 0.116085 |     -0.0941974 |
| baseline   |   2024 | 0.460122 |     -0.107738  |
| model      |   2023 | 0.134104 |     -0.169611  |
| model      |   2024 | 0.311764 |     -0.244107  |

## Limitations

- Historical index membership is not present in the supplied data.
- Industry classification, ST status, and exact exchange price-limit flags are not present.
- The equal-weight universe benchmark is not a tradable index.
- A negative out-of-sample result remains evidence; do not replace it with an in-sample result.
