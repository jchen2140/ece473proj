# Conservative truncation: NYC

**Exploratory counterfactual replay.** The known test-period failure motivated this extension; training-only fitting does not turn this into independent confirmation.

Margin: 1F from the fixed 99th percentile of positive training sensor errors, rounded upward to half degrees (40 days through 2026-05-30). Shared b=300, hypothetical fee=0.05/share. Both parameter selection and margin fitting stop at D-2 of the first test event.

All variants use true open-ended contract bounds and identical coverage-filtered dates. The conservative rule subtracts the fitted margin from the observed running maximum before applying the existing soft/hard downweights. It never uses a test settlement high to quote.

| Maker | Test days | Mean net PnL/day | Worst day | Worst-5% mean PnL | Winner heavily discounted: days | Shares/day |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| forecast_only | 52 | 70.53 | -875.31 | -470.64 | 0 | 9721 |
| current_truncation | 52 | 113.53 | -1419.01 | -863.27 | 1 | 10001 |
| conservative_truncation | 52 | 93.62 | -825.90 | -451.87 | 0 | 9651 |

| Paired PnL difference | Block days | Mean [95% interval] |
| --- | ---: | --- |
| current_truncation - forecast_only | 1 | 43.00 [-39.71, 111.76] |
| conservative_truncation - forecast_only | 1 | 23.09 [8.79, 40.90] |
| conservative_truncation - current_truncation | 1 | -19.91 [-78.25, 53.83] |
| current_truncation - forecast_only | 7 | 43.00 [-20.63, 118.43] |
| conservative_truncation - forecast_only | 7 | 23.09 [9.61, 47.64] |
| conservative_truncation - current_truncation | 7 | -19.91 [-77.81, 35.62] |

Known NYC July 31 failure is retained in daily.csv if it passes the same predeclared coverage rule as every other date. The margin is not a guaranteed bound on future sensor error. Winner-specific loss checks include any downweight of the actual winner; no unconditional 1.2b extra-loss claim is made. Fee income is hypothetical and separated in summary.csv.
