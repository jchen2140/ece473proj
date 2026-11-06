# Conservative truncation: CHI

**Exploratory counterfactual replay.** The known test-period failure motivated this extension; training-only fitting does not turn this into independent confirmation.

Margin: 1F from the fixed 99th percentile of positive training sensor errors, rounded upward to half degrees (42 days through 2026-05-30). Shared b=100, hypothetical fee=0.05/share. Both parameter selection and margin fitting stop at D-2 of the first test event.

All variants use true open-ended contract bounds and identical coverage-filtered dates. The conservative rule subtracts the fitted margin from the observed running maximum before applying the existing soft/hard downweights. It never uses a test settlement high to quote.

| Maker | Test days | Mean net PnL/day | Worst day | Worst-5% mean PnL | Winner heavily discounted: days | Shares/day |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| forecast_only | 69 | 31.83 | -379.49 | -172.79 | 0 | 3324 |
| current_truncation | 69 | 38.23 | -379.49 | -187.44 | 0 | 3279 |
| conservative_truncation | 69 | 33.39 | -379.49 | -171.49 | 0 | 3253 |

| Paired PnL difference | Block days | Mean [95% interval] |
| --- | ---: | --- |
| current_truncation - forecast_only | 1 | 6.40 [-4.61, 17.11] |
| conservative_truncation - forecast_only | 1 | 1.56 [0.67, 2.57] |
| conservative_truncation - current_truncation | 1 | -4.84 [-15.35, 6.01] |
| current_truncation - forecast_only | 7 | 6.40 [0.16, 17.41] |
| conservative_truncation - forecast_only | 7 | 1.56 [0.80, 2.43] |
| conservative_truncation - current_truncation | 7 | -4.84 [-15.94, 1.55] |

Known NYC July 31 failure is retained in daily.csv if it passes the same predeclared coverage rule as every other date. The margin is not a guaranteed bound on future sensor error. Winner-specific loss checks include any downweight of the actual winner; no unconditional 1.2b extra-loss claim is made. Fee income is hypothetical and separated in summary.csv.
