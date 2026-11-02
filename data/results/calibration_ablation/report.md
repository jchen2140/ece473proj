# Controlled calibration experiment

**Exploratory results:** this test period had already been examined before designing the ablation. Confirmation on another period or city is required.

Shared rolling window: 30 days; spread rule: scaled. Chosen once on training days using fully corrected Gaussian log loss, then frozen for all variants. Rolling coefficients update using verified outcomes through D-2, including earlier test-period outcomes as time advances.

All variants use the same 2% uniform mixture and the same eligible test days. The four Gaussian variants form a 2×2 mean/spread experiment; empirical versus uncorrected Gaussian isolates the distribution-form change. Gaussian SD has a common 1.5°F floor.

| Variant | Test days | Brier | Binary log loss | Days with winning-bin probability <1% |
| --- | ---: | ---: | ---: | ---: |
| Empirical GEFS (common smoothing) | 105 | 0.1443 | 0.5002 | 12 |
| Uncorrected Gaussian | 105 | 0.1392 | 0.4823 | 10 |
| Bias correction only | 105 | 0.1224 | 0.3977 | 2 |
| Spread correction only | 105 | 0.1358 | 0.4558 | 4 |
| Bias + spread correction | 105 | 0.1229 | 0.3917 | 1 |

## Paired uncertainty

Negative differences favor the first variant. Weekly moving-block intervals assess sensitivity to dependence across consecutive weather days; six bins of each day are kept together.

| First − second | Metric | Block length | Difference [95% interval] |
| --- | --- | ---: | --- |
| gaussian_uncorrected − empirical_smoothed | brier | 1 | -0.0050 [-0.0095, -0.0005] |
| gaussian_uncorrected − empirical_smoothed | log_loss | 1 | -0.0179 [-0.0396, 0.0021] |
| bias_only − gaussian_uncorrected | brier | 1 | -0.0168 [-0.0278, -0.0042] |
| bias_only − gaussian_uncorrected | log_loss | 1 | -0.0846 [-0.1312, -0.0370] |
| spread_only − gaussian_uncorrected | brier | 1 | -0.0034 [-0.0064, -0.0006] |
| spread_only − gaussian_uncorrected | log_loss | 1 | -0.0266 [-0.0428, -0.0119] |
| bias_and_spread − gaussian_uncorrected | brier | 1 | -0.0164 [-0.0257, -0.0057] |
| bias_and_spread − gaussian_uncorrected | log_loss | 1 | -0.0906 [-0.1362, -0.0476] |
| bias_and_spread − bias_only | brier | 1 | 0.0004 [-0.0031, 0.0038] |
| bias_and_spread − bias_only | log_loss | 1 | -0.0060 [-0.0229, 0.0080] |
| bias_and_spread − spread_only | brier | 1 | -0.0130 [-0.0211, -0.0039] |
| bias_and_spread − spread_only | log_loss | 1 | -0.0641 [-0.0969, -0.0315] |
| gaussian_uncorrected − empirical_smoothed | brier | 7 | -0.0050 [-0.0086, -0.0020] |
| gaussian_uncorrected − empirical_smoothed | log_loss | 7 | -0.0179 [-0.0371, 0.0020] |
| bias_only − gaussian_uncorrected | brier | 7 | -0.0168 [-0.0300, 0.0005] |
| bias_only − gaussian_uncorrected | log_loss | 7 | -0.0846 [-0.1589, -0.0016] |
| spread_only − gaussian_uncorrected | brier | 7 | -0.0034 [-0.0070, 0.0001] |
| spread_only − gaussian_uncorrected | log_loss | 7 | -0.0266 [-0.0521, -0.0027] |
| bias_and_spread − gaussian_uncorrected | brier | 7 | -0.0164 [-0.0308, -0.0004] |
| bias_and_spread − gaussian_uncorrected | log_loss | 7 | -0.0906 [-0.1649, -0.0110] |
| bias_and_spread − bias_only | brier | 7 | 0.0004 [-0.0031, 0.0025] |
| bias_and_spread − bias_only | log_loss | 7 | -0.0060 [-0.0258, 0.0058] |
| bias_and_spread − spread_only | brier | 7 | -0.0130 [-0.0240, 0.0010] |
| bias_and_spread − spread_only | log_loss | 7 | -0.0641 [-0.1175, -0.0049] |

These forecast variants are exported in `probabilities.csv` with model parameters and calibration-observation provenance. Forecast parameters describe the underlying distribution before the common uniform mixture. The 1% threshold is a descriptive diagnostic, not a trading rule.
