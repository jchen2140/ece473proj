# Conservative truncation: LAX

**Exploratory counterfactual replay.** The known test-period failure motivated this extension; training-only fitting does not turn this into independent confirmation.

Margin: 52F from the fixed 99th percentile of positive training sensor errors, rounded upward to half degrees (27 days through 2026-05-31). Shared b=300, hypothetical fee=0.05/share. Both parameter selection and margin fitting stop at D-2 of the first test event.

All variants use true open-ended contract bounds and identical coverage-filtered dates. The conservative rule subtracts the fitted margin from the observed running maximum before applying the existing soft/hard downweights. It never uses a test settlement high to quote.

| Maker | Test days | Mean net PnL/day | Worst day | Worst-5% mean PnL | Winner heavily discounted: days | Shares/day |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| forecast_only | 23 | 222.10 | -138.36 | -123.40 | 0 | 10365 |
| current_truncation | 23 | 281.25 | -107.29 | -88.33 | 0 | 10662 |
| conservative_truncation | 23 | 222.10 | -138.36 | -123.40 | 0 | 10365 |

| Paired PnL difference | Block days | Mean [95% interval] |
| --- | ---: | --- |
| current_truncation - forecast_only | 1 | 59.15 [-20.63, 149.96] |
| conservative_truncation - forecast_only | 1 | 0.00 [0.00, 0.00] |
| conservative_truncation - current_truncation | 1 | -59.15 [-149.96, 20.63] |
| current_truncation - forecast_only | 7 | 59.15 [-17.37, 131.78] |
| conservative_truncation - forecast_only | 7 | 0.00 [0.00, 0.00] |
| conservative_truncation - current_truncation | 7 | -59.15 [-131.78, 17.37] |

Known NYC July 31 failure is retained in daily.csv if it passes the same predeclared coverage rule as every other date. The margin is not a guaranteed bound on future sensor error. Winner-specific loss checks include any downweight of the actual winner; no unconditional 1.2b extra-loss claim is made. Fee income is hypothetical and separated in summary.csv.
