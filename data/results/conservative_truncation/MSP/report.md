# Conservative truncation: MSP

**Exploratory counterfactual replay.** The known test-period failure motivated this extension; training-only fitting does not turn this into independent confirmation.

Margin: 1F from the fixed 99th percentile of positive training sensor errors, rounded upward to half degrees (42 days through 2026-05-30). Shared b=30, hypothetical fee=0.08/share. Both parameter selection and margin fitting stop at D-2 of the first test event.

All variants use true open-ended contract bounds and identical coverage-filtered dates. The conservative rule subtracts the fitted margin from the observed running maximum before applying the existing soft/hard downweights. It never uses a test settlement high to quote.

| Maker | Test days | Mean net PnL/day | Worst day | Worst-5% mean PnL | Winner heavily discounted: days | Shares/day |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| forecast_only | 60 | -6.23 | -71.99 | -56.86 | 0 | 469 |
| current_truncation | 60 | -4.04 | -84.42 | -71.41 | 0 | 460 |
| conservative_truncation | 60 | -5.35 | -71.99 | -56.84 | 0 | 448 |

| Paired PnL difference | Block days | Mean [95% interval] |
| --- | ---: | --- |
| current_truncation - forecast_only | 1 | 2.19 [-1.29, 5.32] |
| conservative_truncation - forecast_only | 1 | 0.88 [0.42, 1.39] |
| conservative_truncation - current_truncation | 1 | -1.31 [-4.46, 2.05] |
| current_truncation - forecast_only | 7 | 2.19 [-1.94, 6.56] |
| conservative_truncation - forecast_only | 7 | 0.88 [0.46, 1.27] |
| conservative_truncation - current_truncation | 7 | -1.31 [-5.70, 2.85] |

Known NYC July 31 failure is retained in daily.csv if it passes the same predeclared coverage rule as every other date. The margin is not a guaranteed bound on future sensor error. Winner-specific loss checks include any downweight of the actual winner; no unconditional 1.2b extra-loss claim is made. Fee income is hypothetical and separated in summary.csv.
