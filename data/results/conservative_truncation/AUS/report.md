# Conservative truncation: AUS

**Exploratory counterfactual replay.** The known test-period failure motivated this extension; training-only fitting does not turn this into independent confirmation.

Margin: 1F from the fixed 99th percentile of positive training sensor errors, rounded upward to half degrees (23 days through 2026-05-29). Shared b=30, hypothetical fee=0.05/share. Both parameter selection and margin fitting stop at D-2 of the first test event.

All variants use true open-ended contract bounds and identical coverage-filtered dates. The conservative rule subtracts the fitted margin from the observed running maximum before applying the existing soft/hard downweights. It never uses a test settlement high to quote.

| Maker | Test days | Mean net PnL/day | Worst day | Worst-5% mean PnL | Winner heavily discounted: days | Shares/day |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| forecast_only | 31 | 7.74 | -22.16 | -21.49 | 0 | 931 |
| current_truncation | 31 | 9.37 | -16.27 | -15.53 | 0 | 930 |
| conservative_truncation | 31 | 8.39 | -23.50 | -21.40 | 0 | 915 |

| Paired PnL difference | Block days | Mean [95% interval] |
| --- | ---: | --- |
| current_truncation - forecast_only | 1 | 1.63 [-3.61, 6.99] |
| conservative_truncation - forecast_only | 1 | 0.65 [0.25, 1.20] |
| conservative_truncation - current_truncation | 1 | -0.98 [-6.18, 4.13] |
| current_truncation - forecast_only | 7 | 1.63 [-4.04, 8.45] |
| conservative_truncation - forecast_only | 7 | 0.65 [0.23, 1.08] |
| conservative_truncation - current_truncation | 7 | -0.98 [-7.57, 4.56] |

Known NYC July 31 failure is retained in daily.csv if it passes the same predeclared coverage rule as every other date. The margin is not a guaranteed bound on future sensor error. Winner-specific loss checks include any downweight of the actual winner; no unconditional 1.2b extra-loss claim is made. Fee income is hypothetical and separated in summary.csv.
