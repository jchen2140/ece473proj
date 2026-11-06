# Conservative live truncation: NYC pilot

## Main result

A **1°F margin selected only from available training observations** avoids heavily discounting the winning bin on the known July 31 NYC failure day. That day's simulated loss falls from **$1,419.01 to $149.47**. The day is retained, not removed.

Across **52 identical coverage-qualified test dates**, conservative truncation earns less on average than current truncation, but has a better observed worst day and worst-5% average:

| Maker | Mean net PnL/day | Worst day | Mean PnL on 3 worst days | Heavily discounted winner: days | Shares/day |
| --- | ---: | ---: | ---: | ---: | ---: |
| Forecast only | $70.53 | -$875.31 | -$470.64 | 0 | 9,721 |
| Current truncation | $113.53 | -$1,419.01 | -$863.27 | 1 | 10,001 |
| Conservative truncation | $93.62 | -$825.90 | -$451.87 | 0 | 9,651 |

These use common **b=300** and a hypothetical maker fee of **$0.05/share**. The conservative-minus-current mean difference is -$19.91/day, with a seven-day calendar-block interval of -$77.81 to $35.62. It does not establish that either has higher expected PnL. Conservative-minus-forecast-only is +$23.09/day, with a weekly-block interval of $9.61 to $47.64, under these replay assumptions.

The observed tail comparison is descriptive and based on just three days per maker. It is not proof of lower population tail risk.

## Training-only rule

For each eligible training day, measure:

```text
sensor error = maximum one-minute station reading - official settlement high
```

Keep the positive part of that error. The margin is its empirical nearest-rank 99th percentile, rounded **up** to half degrees. The 99% quantile, 80% minimum daily minute coverage, and rounding rule were fixed before this run and were not chosen by comparing test PnL.

NYC has **40 eligible training dates through May 30**, producing a 1°F margin. With only 40 dates, the nearest-rank 99th percentile equals the largest positive training error; it should not be interpreted as an accurately estimated 99% future coverage guarantee.

At each quote, the conservative maker applies the original soft/hard truncation logic to:

```text
adjusted running maximum = observed running maximum - fitted margin
```

It applies weight 0.3 after that adjusted maximum crosses a bin's upper edge, and weight 0.001 after it clears the edge by another 1°F. Original live truncation is identical with a zero margin. The same five-minute feed-delay proxy is retained.

Common b and fee are chosen using the **forecast-only** maker's training PnL. All variants then receive those same settings. Both setting selection and margin fitting stop on **May 30**, because the first test event is June 1 and May 31's high is not final at the previous-day noon opening cutoff. No test settlement temperature enters any quote decision.

## Data and comparison corrections

- Raw minute observations and the NYC replay trade cache are absent from Git, so this run rebuilds them using `collect.realtime`. NYC reuses the committed Kalshi flow table rather than re-querying all trades.
- At least 80% of the day's expected 1,440 distinct minute timestamps must be present. Missing official highs and absent trade flow also exclude a date. `sensor_audit.csv` preserves every reason. This leaves 40 fitting dates and 52 common evaluation dates, rather than all 74 June–August test dates. Missingness may be systematic, so do not generalize these estimates to excluded dates without further study.
- Every maker uses **true infinite bounds for open-ended tails**. `forecast.public.load_city` creates finite representative tail values for implied medians; those are not valid knockout boundaries. The new experiment reconstructs true bounds from each market's strike metadata for all three makers. Its output therefore is not an exact reproduction of the older adaptive-maker table.
- The known July 31 case passes the coverage rule with 100% minute coverage. Official high is 85°F; sensor maximum is 87°F. Its winner is assigned weight 0.001 by current truncation and 0.3 by the conservative rule. Thus the rule reduces, but does not eliminate, the mistake.
- Sensitivity intervals resample whole dates. Weekly resampling uses the full calendar with missing-date masks; it does not connect gaps as if they were adjacent, discard isolated valid dates, or wrap the last date back to the first.

## Loss accounting

The implementation reuses `market.makers.replay`; its only API extension is an optional tilt callback. Default behavior is unchanged.

For a fixed-b run and actual winner w, every run checks the conservative pre-fee loss bound:

```text
loss <= b * log(1 / initial_probability_w) + b * log(1 / final_tilt_weight_w)
```

This drops the nonpositive cost-function jump terms, so it remains conservative even when the sensor mistakenly discounts the winner. It does **not** make an unconditional claim that extra loss is capped at 1.2b. A margin estimated from historical records can still be exceeded in the future.

Fee income and trading PnL are separate fields in `summary.csv`. Trade arrivals are replayed counterfactually and may differ from what traders would do against these new quotes. Five-minute publication delay is a scenario proxy, not verified historical feed timing.

## Reproduce on the Milestone III code

```bash
python -m pip install -r requirements.txt
python -m collect.realtime --cities NYC
python -m market.conservative --cities NYC
python -m market.plot_conservative --city NYC
python -m unittest discover -s tests -t .
```

Other cities are supported by `--cities`, but require their corresponding raw minute and trade caches. This execution covers **NYC only**, not all six cities.

Outputs are isolated in `data/results/conservative_truncation/NYC/`:

- `sensor_audit.csv`: temperature errors, minute coverage, and eligibility.
- `training_settings.csv`: common forecast-only training grid.
- `daily.csv`: all three makers, actual winner downweights, cash accounting, and loss bounds.
- `summary.csv` / `paired_differences.csv`: point estimates and day/weekly intervals.
- `manifest.json`: training margin, availability cutoff, shared settings, and input hashes.
- `report.md` / `comparison.png`: generated summary and figure.

## Status and next confirmation

The test-period failure motivated this extension, so even training-only parameter fitting makes this an **exploratory robustness experiment**, not an independent confirmation. The next step is to freeze the rule and test it in another city or an untouched period. No claim of a guaranteed sensor-error bound, optimal margin, or realized trading profit is supported.
