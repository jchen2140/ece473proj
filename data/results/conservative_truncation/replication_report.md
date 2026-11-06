# Frozen-rule conservative-truncation replication

CHI (Chicago Midway) was selected before inspecting its conservative replay results. The 99% training-error quantile, upward half-degree rounding, 80% minute coverage and five-minute delay were frozen. Each city fits its own numerical margin and common liquidity/fee setting using available training data only. No strategy change was made after Chicago results.

**Scope:** out-of-city replication of a frozen procedure, not a fully blind external validation. The team had previously examined unguarded Chicago aggregate results. The NYC failure motivated the design.

The margin column is the city-fitted value; only conservative truncation applies it. Winner counts refer to the heavy 0.001 downweight, not the softer 0.3 discount.

| City | Maker | Margin | Test days | Mean net PnL/day | Worst-5% mean PnL | Worst day | Heavily discounted winners | Shares/day |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| NYC | forecast_only | 1F | 52 | 70.53 | -470.64 | -875.31 | 0 | 9721 |
| NYC | current_truncation | 1F | 52 | 113.53 | -863.27 | -1419.01 | 1 | 10001 |
| NYC | conservative_truncation | 1F | 52 | 93.62 | -451.87 | -825.90 | 0 | 9651 |
| CHI | forecast_only | 1F | 69 | 31.83 | -172.79 | -379.49 | 0 | 3324 |
| CHI | current_truncation | 1F | 69 | 38.23 | -187.44 | -379.49 | 0 | 3279 |
| CHI | conservative_truncation | 1F | 69 | 33.39 | -171.49 | -379.49 | 0 | 3253 |

## Paired weekly-block uncertainty

Positive differences favor conservative truncation. Each tail statistic is recomputed inside each shared-date resample. These are percentile intervals conditional on fitted strategies; very small tail samples limit precision. Missing-date masks are preserved.

| City | Conservative minus | Metric | Difference [95% interval] |
| --- | --- | --- | --- |
| NYC | current_truncation | mean_net_pnl | -19.91 [-77.81, 35.62] |
| NYC | current_truncation | worst_5pct_mean_pnl | 411.40 [-37.73, 1141.14] |
| NYC | forecast_only | mean_net_pnl | 23.09 [9.61, 47.64] |
| NYC | forecast_only | worst_5pct_mean_pnl | 18.77 [0.00, 49.41] |
| CHI | current_truncation | mean_net_pnl | -4.84 [-15.94, 1.55] |
| CHI | current_truncation | worst_5pct_mean_pnl | 15.95 [-16.87, 49.51] |
| CHI | forecast_only | mean_net_pnl | 1.56 [0.80, 2.43] |
| CHI | forecast_only | worst_5pct_mean_pnl | 1.30 [0.00, 3.43] |

## Interpretation by city

- NYC: observed tail-mean change +411.40/day; average net-PnL change -19.91/day versus current truncation. Heavy winner discounts: 1 to 0.
- CHI: observed tail-mean change +15.95/day; average net-PnL change -4.84/day versus current truncation. Heavy winner discounts: 0 to 0.

Chicago has 69 common test dates and four dates in each worst-5% tail. Its margin is 1F from 42 training dates through May 30, with common b=100 and hypothetical fee=$0.05/share. Five of the 74 test dates fail the fixed minute-coverage rule. Unlike NYC, Chicago has no heavily discounted winning bins under either truncation rule, and its worst day is unchanged. It therefore does not independently reproduce the rare sensor-error failure or demonstrate that the safeguard prevents it. The observed tail improvement is small and its interval includes zero.

No city was dropped or replaced based on results. No pooled two-city significance claim is made. Report gains, losses and intervals together. Different city-specific b and fees mean dollar levels should be compared within cities, not interpreted as a common-capital cross-city ranking.

Net PnL includes hypothetical maker fee income. Counterfactual replay assumes observed traders would offer the same trades under different quotes. Live publication delay is a scenario proxy. The empirical training quantile is not a guaranteed bound on future sensor errors.
