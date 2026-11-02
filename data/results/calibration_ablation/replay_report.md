# Calibration and market-making experiment

**Exploratory counterfactual replay, not realized trading performance.** The current test period was already inspected before this experiment was designed.

Main fee: $0.05/share, reused from the existing train-selected NOAA replay. Equal opening loss budget: $300/event. Fixed-b reference: 300. No settings were optimized against these new test results. Fee-free runs are separate replays with different fills, not subtraction of fees from a fee-charging run.

The fee is a hypothetical charge collected by this maker, not an assertion about Kalshi exchange fees. Quotes halt at noon on the target day. All variants face the same observed arrivals; traders may behave differently in reality.

## equal_loss_budget

| Variant | Opening Brier | Trading PnL/day | Fee income/day | Net PnL/day [95% day interval] | Worst-5% mean PnL | Shares/day | Mean loss bound |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| Bias + spread correction | 0.1229 | $-56.30 | $71.05 | $14.75 [$2.09, $26.47] | $-103.46 | 1421 | $300.00 |
| Bias correction only | 0.1224 | $-51.89 | $63.65 | $11.75 [$-1.38, $24.08] | $-124.00 | 1273 | $300.00 |
| Empirical GEFS (common smoothing) | 0.1443 | $-75.50 | $66.32 | $-9.18 [$-24.58, $4.67] | $-199.00 | 1326 | $300.00 |
| Uncorrected Gaussian | 0.1392 | $-79.62 | $72.58 | $-7.04 [$-22.65, $7.09] | $-198.56 | 1452 | $300.00 |
| Kalshi midpoint (common smoothing) | 0.1094 | $-40.82 | $68.61 | $27.79 [$18.77, $37.69] | $-65.35 | 1372 | $300.00 |
| Spread correction only | 0.1358 | $-80.98 | $80.19 | $-0.80 [$-15.43, $13.08] | $-189.15 | 1604 | $300.00 |
## fixed_b

| Variant | Opening Brier | Trading PnL/day | Fee income/day | Net PnL/day [95% day interval] | Worst-5% mean PnL | Shares/day | Mean loss bound |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| Bias + spread correction | 0.1229 | $-218.78 | $235.65 | $16.87 [$-28.78, $63.20] | $-448.09 | 4713 | $1303.67 |
| Bias correction only | 0.1224 | $-228.85 | $234.62 | $5.77 [$-54.90, $59.53] | $-694.01 | 4692 | $1454.90 |
| Empirical GEFS (common smoothing) | 0.1443 | $-389.27 | $283.58 | $-105.69 [$-190.51, $-33.66] | $-1173.28 | 5672 | $1606.83 |
| Uncorrected Gaussian | 0.1392 | $-364.20 | $273.24 | $-90.96 [$-168.08, $-23.19] | $-1116.93 | 5465 | $1444.73 |
| Kalshi midpoint (common smoothing) | 0.1094 | $-147.68 | $208.30 | $60.63 [$29.98, $93.52] | $-257.05 | 4166 | $1140.49 |
| Spread correction only | 0.1358 | $-320.57 | $265.16 | $-55.41 [$-119.65, $1.38] | $-885.46 | 5303 | $1269.27 |

CSV outputs include zero-fee sensitivity, paired PnL differences, and seven-day moving-block intervals. Positive paired PnL differences favor the first variant. Worst-5% PnL averages the six worst days out of 105; a less negative value is better. This small tail sample is noisy.

Equal loss budgets control the worst-case opening bound, not equal turnover or equal realized risk. Within fixed-b runs only b and fee are shared; their loss bounds differ by forecast sharpness.

`winner_probability_diagnostics.csv` conditions on the **realized** winning bin and is descriptive only; its groups cannot be used as an available-at-cutoff trading signal.
