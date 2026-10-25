# Data notes: full-window run (Milestone 1, week 3)

What the data forced, what changed from the plan, and why. Checked October 24–25, 2026.

## Window and split (as planned, no change)

| | Dates | Days | Contracts |
| --- | --- | ---: | ---: |
| Whole window | Aug 14, 2025 – Aug 13, 2026 | 365 | 2,190 |
| Train | Aug 14, 2025 – Apr 30, 2026 | 260 | 1,560 |
| Test | May 1 – Aug 13, 2026 | 105 | 630 |

All dates live in `config.py`. All 365 days were collected and none were skipped (`data/skipped_days.csv` is empty). The fallback window (Feb 1 – Aug 13, 2026) was not needed.

**Scope change from the one-pager:** NYC Central Park only for Milestone 1 (the one-pager said "certain cities"). What decides whether one model beats another is the number of days, and the code is built around one station. Chicago Midway is planned for Milestone 2.

**Limitation:** the test period is late spring and summer only. The train period covers late summer, fall, winter and early spring.

## Why the window ends Aug 13, 2026

Checked with `GET /events/{event_ticker}` and each market's `rules_primary`:

- KXHIGHNY-25AUG14 (first day) and KXHIGHNY-26AUG13 (last day): rules name Central Park and the National Weather Service Climatological Report.
- KXHIGHNY-26AUG14 onward: rules say "according to The Weather Company". These settle on a different source and are excluded.

The collector still checks every day's rules and would skip a non-NWS day.

## Historical-markets query: worked as is

The concern was that `historical/markets?event_ticker=...` ignored the filter (a web fetcher returned KXHIGHNY-25OCT15 for other dates). Querying the API directly, KXHIGHNY-25JAN15, 25APR15, 25AUG14 and 25DEC03 each returned their own six markets. The fetcher was caching. No change was needed. The collector still skips a day if any returned market belongs to another event.

## Blank settlement temperatures (68 train days)

Kalshi left `expiration_value` (the reported high) blank on every bin for **68 events, Oct 31, 2025 – Jan 12, 2026**. Each of those events still has exactly one YES bin.

- **Bin outcomes (YES/NO) always come from Kalshi.** Every score in the results tables uses only these.
- For those 68 days, `kalshi_settlement_high_f` is filled with the NOAA GHCN Central Park high for that date (rounded half up). It is used only if it falls inside Kalshi's winning bin, otherwise the day would be skipped. All 68 passed. The column `settlement_high_source` is `ghcn_fallback` on those rows and `kalshi` everywhere else.
- **No test day uses the fallback.**
- On the 297 days where Kalshi did report the high, the rounded GHCN high matches it on **all 297** (`data/observation_crosscheck.csv`). That supports using GHCN where Kalshi is blank.
- One more case: on **Jun 23, 2026**, five bins reported 71°F and one was blank. The collector now uses the value when every non-blank bin agrees, and skips the day if they disagree. GHCN also says 71°F.

## Market quotes

12 contracts (all six bins on **Nov 3 and Nov 4, 2025**, both train days) have no bid/ask midpoint at the noon cutoff. They stay missing. Every test contract has a quote, so the main comparison uses all 630 test contracts.

Bid/ask midpoints are quotes, not prices anyone traded at.

## Collector changes

- **Skip instead of crash:** a day that fails a strict check raises `SkipDay(reason)`, is written to `data/skipped_days.csv`, and the run moves on. Reasons: `wrong_bin_count`, `non_nws_rules`, `unexpected_event`, `outcome_count`, `missing_settlement_high`, `inconsistent_settlement_high`, `ghcn_fallback_inconsistent`, `gefs_missing`, `gefs_invalid`, `gefs_after_cutoff`, `outcome_inconsistent`, `bins_not_partition`, `network_error`. The checks themselves are unchanged.
- **Cheap reruns:** each finished day is saved to `data/days/YYYY-MM-DD.json` (gitignored). A rerun loads those and fetches only what is missing. Network failures are not saved, so a rerun retries them. Rebuilding all 365 days from the saved files takes about 36 seconds.
- **Retries:** `fetch` now also retries dropped connections and socket timeouts. On Python 3.9, `socket.timeout` is not a `TimeoutError`, so it was not retried before.

## Run time (timed test and full window)

Each day is 310 GEFS files (31 members × 10 forecast times). Each file is the full global 0.25° temperature field, about 0.76 MB, and it cannot be cut down to one grid point before downloading. So each day is about 160–230 MB, and the year was about **67 GB** (in `data/raw/gefs/`, gitignored). Speed is limited by download bandwidth, not the code.

| Run | Speed |
| --- | --- |
| 1 day, `--workers 6` (the old default) | 111 s/day → about 11 hours for 365 days |
| 1 day, `--workers 32` | ~37 s/day → about 3.75 hours for 365 days |
| Timed test, `--start 2026-07-14 --days 31 --workers 32`, sharing bandwidth with four other runs | 30 days in 1,621 s (27 min); the 31st hit a timeout and was retried later |
| Full window: five date ranges in parallel, 32 workers each | about 1 h 35 min wall time (22:48 – 00:24), then a 36 s assembly pass |

The 0.5° GEFS product is about 3× smaller per file (0.24 MB) and would be faster. It was not used because it would change the grid point and the pilot's method.

## Evaluation conventions (evaluate.py)

- Test days only for the main table. Every model is scored on the same contracts.
- Brier and log loss are averaged per binary contract. Log loss clips probabilities to [1e-6, 1 − 1e-6] (same as `milestone1.score`).
- 95% intervals: resample the 105 test days with replacement 1,000 times (seed 473) and take the 2.5th and 97.5th percentiles. Paired differences are computed inside each resample, so they get their own intervals.
- **Log-loss caveat:** on 12 of 630 test contracts, all 31 members missed the bin that won, so the raw ensemble gave it probability 0. Each of those costs −ln(1e-6) ≈ 13.8 in log loss. Together they are about 0.26 of the ensemble's 0.66 mean. The results tables show the same comparison with clips of 1e-3 and 1e-2. Brier is not affected.
- **Persistence baseline:** forecasts are made at noon the day before, when yesterday's high is not final. So it uses the high from two days earlier (GHCN) plus the 1991–2020 distribution of two-day changes within ±15 calendar days.
- Climatology, persistence and the raw ensemble have no parameters fit on 2025–26 data. A whole-window table is included as a supplement. The headline uses test days only because the train days will be used for bias/spread fitting.

## Note for the bias/spread adjustment (Jessie's task)

Mean (observed high − ensemble mean): **+1.30°F on train days** (ensemble too cold) and **−2.11°F on test days** (ensemble too warm in summer). A single constant fit on Aug–Apr would shift summer forecasts the wrong way, so the adjustment probably needs a seasonal or recent-days training window. Per-day ensemble means can be computed from `data/ensemble_daily_maxima.csv`.

## Other differences from the July pilot

- `data/historical_tmax.csv` was re-downloaded on Oct 25, 2026: 13,079 accepted rows, up from 13,077. The 1991–2020 training rows used by climatology are unchanged.
- `milestone1.py` now scores test days only (from `config.py`), so `data/evaluation.json` and `data/calibration.csv` describe the 105 test days, not the 7 pilot days.
