# Conservative truncation: frozen-rule city replication

## Question

Does subtracting a training-estimated sensor-error margin before discounting temperature bins
produce the same risk/return tradeoff outside the NYC sample that motivated the idea?

Chicago Midway (CHI/KMDW) was chosen before inspecting the new conservative replay outputs.
The saved protocol in `data/results/conservative_truncation/cross_city_protocol.json` records
the choice, hypotheses, settings and SHA-256 hashes of the core strategy implementation.
`market.cross_city` verifies those hashes before running the replication. Earlier aggregate
results for unguarded Chicago truncation were already visible in the repository, so this is
an out-of-city replication of a frozen procedure, not fully blind external validation.

## Frozen procedure

- Use April–May training data and June 1–August 13 test dates, subject to identical eligibility
  rules for all makers: at least 80% distinct daily minute coverage, an official settlement
  temperature, and observed trade flow.
- Estimate each city's margin from the nearest-rank 99th percentile of positive training
  sensor-minus-official errors, rounded upward to 0.5°F. Stop training at D-2 relative to
  the first eligible test event. With fewer than 100 training days this percentile is the
  maximum training error, not evidence of 99% future coverage.
- Choose each city's common liquidity parameter and hypothetical fee using forecast-only
  training PnL. Apply the same settings to forecast only, current live truncation and
  conservative live truncation. Do not tune separate settings for favorable test results.
- Keep the five-minute observation-delay proxy and true open-ended contract boundaries.
- Report Chicago even if its fitted margin is zero or conservative truncation performs worse.

## Comparisons and uncertainty

The primary replication comparison is conservative minus current truncation, for average net
PnL and the average of each maker's worst 5% of test days. Each tail consists of the lowest
`ceil(0.05*n)` daily PnL values. A positive difference means conservative truncation improves
that statistic. The two makers' worst days can differ, so the tail of their paired PnL
differences is not the quantity of interest.

The report includes 1,000 paired daily and seven-calendar-day block bootstrap resamples.
Every draw uses shared dates for all makers and recomputes each maker's own tail. Calendar
blocks preserve missing-date masks. Intervals condition on the training-fitted strategies;
they do not include model-selection uncertainty or prove extreme-loss safety. Tail estimates
are particularly imprecise because they contain very few dates.

Report each city separately. Different fitted liquidity and fee settings make absolute
cross-city dollar levels unsuitable for a common-capital ranking. No pooled two-city
significance claim is made.

## Reproduce

From the Milestone III layout at base revision `8f694e63b88f1b6292ec890f799a62e45ad7611b`,
with the delivered implementation applied:

```bash
python -m collect.realtime --cities NYC CHI
python -m market.conservative --cities NYC
python -m market.cross_city
python -m market.plot_cross_city
python -m unittest discover -s tests -t .
```

The frozen protocol must be present at the path above. Large raw temperature and trade
caches remain ignored by Git. Input hashes are stored in each city's `manifest.json`.
The root `replication_report.md`, `replication_summary.csv`,
`replication_paired_intervals.csv` and `replication_comparison.png` contain the combined
outputs; city subdirectories retain the underlying daily replay, coverage audit and training
grid. The NYC pilot's detailed interpretation remains in `docs/CONSERVATIVE_TRUNCATION.md`.

These are counterfactual replay results with hypothetical fee revenue, not realized profits.
Observed trade arrivals may change under different maker quotes, minute-sensor readings can
disagree with official settlement highs, and publication timing is approximated.
