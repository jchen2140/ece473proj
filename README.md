# NYC Central Park weather-market pilot

This project pairs settled Kalshi daily-high contracts with archived NOAA GEFS ensemble forecasts for Central Park. The original pilot sample was **July 20–26, 2026** (seven days, a pipeline check only); the collector now defaults to the one-year window in `config.py`.

For the first two weeks' task-by-task mapping and the complete workflow, see [MILESTONE1_COVERAGE.md](MILESTONE1_COVERAGE.md). Run `python3 milestone1.py` after collection to add historical station temperatures, a seasonal climatology baseline, and Brier/log-loss/calibration evaluation.

The dedicated [week 2 baseline guide](CLIMATOLOGY_BASELINE.md) documents the historical records and model. Run `python3 climatology.py` to produce the 1991–2020 training set, post-training records, coverage checks, and seasonal summaries. For a probability query, use `python3 climatology.py --date 2026-07-20 --lower 79 --upper 80`.

## Full-window run (Milestone 1, week 3)

`config.py` sets the study window (Aug 14, 2025 – Aug 13, 2026, every contract on NWS/Central Park rules) and the date split: train Aug 14, 2025 – Apr 30, 2026, test May 1 – Aug 13, 2026. With no arguments, `collect_nyc.py` now collects that whole window. See [DATA_NOTES.md](DATA_NOTES.md) for what the data forced and the plan changes.

```bash
.venv/bin/python collect_nyc.py --workers 32   # resumable; finished days are saved in data/days/
.venv/bin/python milestone1.py                 # climatology for every row; point scores on test days
.venv/bin/python evaluate.py                   # test-only scores with day-bootstrap CIs -> data/results/
python3 inspect_nyc.py                         # offline consistency checks
.venv/bin/python -m unittest test_milestone1.py test_climatology.py test_evaluation.py
```

- **Skipped days:** a day that fails a strict check (not six settled bins, rules not naming Central Park/NWS, a market from another event, missing or late GEFS files, inconsistent outcome) is written to `data/skipped_days.csv` with a reason instead of stopping the run. Network failures are not cached, so rerunning retries them.
- **Blank settlement highs:** some 2025 events settled every bin but have an empty `expiration_value`. For those days `kalshi_settlement_high_f` is the GHCN Central Park high, used only if it falls in Kalshi's winning bin; `settlement_high_source` says `ghcn_fallback` instead of `kalshi`.
- **`evaluate.py`:** scores the raw ensemble, climatology, persistence (the high from two days before plus 1991–2020 two-day changes; yesterday's high is not final at the noon cutoff) and the market midpoint on identical test-day rows. 95% intervals resample whole test days 1,000 times; paired differences (ensemble − climatology, ensemble − market, ensemble − persistence) get their own intervals. Outputs: `data/results/results_table.csv`, `paired_differences.csv`, `results_tables.md`, `scored_rows.csv`, `summary.json`.

## Run

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python collect_nyc.py
```

For a one-day check, append `--start 2026-07-20 --days 1`. Downloads are cached and reruns reuse them. `--start YYYY-MM-DD --days N` changes the period, but the collector skips (and logs) days with contracts whose rules do not explicitly name Central Park and the NWS. No Kalshi login or AWS account is needed.

Git includes the extracted datasets, Kalshi responses, and NOAA source manifests. Large raw GEFS messages, the downloadable GHCN station/history files, and the virtual environment are ignored. A fresh clone can recreate those caches by running the workflow in `MILESTONE1_COVERAGE.md`.

## Outputs

| File | Contents |
| --- | --- |
| `data/matched.csv` | One row per contract: bin, ensemble probability, previous-day market midpoint, result |
| `data/markets.csv` | Contract metadata, rules, settlement high, bid/ask and timestamp |
| `data/ensemble_temperatures.csv` | 31 members × 10 forecast times × number of days; nearest model grid point |
| `data/ensemble_daily_maxima.csv` | Approximate daily high for each member |
| `data/validation.json` | Completeness checks, descriptive Brier scores, limitations |
| `data/daily_summary.csv` | One row per day: settlement high, ensemble mean/range, winning bin |
| `data/raw/` | Original API responses and source URLs; cached GRIB temperature messages and retrieval metadata |
| `data/historical_tmax.csv` | Quality-screened NOAA GHCN Central Park daily highs from 1991 onward |
| `data/matched_with_climatology.csv` | Existing matched data plus seasonal climatology probabilities |
| `data/climatology_training_summary.csv` | Historical sample counts and date bounds per target day |
| `data/evaluation.json` | Ensemble, climatology, and market scores on identical contract rows |
| `data/calibration.csv` | Reliability-bin predictions, outcomes, and counts for all three forecasts |
| `data/observation_crosscheck.csv` | Independent GHCN daily-high comparison against Kalshi's settlement high |

## Matching choices

- **Station:** KNYC, New York City/Central Park. Latitude and longitude are retrieved from the NWS station API and saved in `data/raw/station_KNYC.json`.
- **Series:** `KXHIGHNY`; individual contracts come from the historical and live market endpoints, with pagination and deduplication.
- **Prediction time:** noon in `America/New_York` on the previous day (16:00 UTC for this July sample).
- **Market probability proxy:** YES bid/ask midpoint at the latest hourly candle ending no later than prediction time, using at most two hours of candle history. Its age is visible in the timestamp column. It is not an executable trading price. Missing quotes remain missing.
- **Model:** GEFS, previous-day 06 UTC run, 0.25-degree `pgrb2sp25` product. Control plus 30 perturbations are equally weighted. The collector fetches only the TMP/2 m GRIB byte range, not the entire global multi-variable file. Source-object modification times must precede the prediction cutoff.
- **Spatial extraction:** geometric nearest grid point, with grid coordinates and distance saved. This is a forecast for a grid cell near Central Park, not an exact station forecast. No elevation, coastal, or urban bias correction is applied.
- **Climate day:** midnight to midnight local standard time, i.e. 05:00–05:00 UTC year-round for NYC. In July this corresponds to 01:00–01:00 civil time.
- **Daily high approximation:** three-hourly temperatures at 06, 09, 12, 15, 18, 21, 00 and 03 UTC, plus linearly interpolated values at the bounding 05 UTC times. Take the maximum separately for each member. This can miss between-sample maxima. The true continuous model maximum is not recovered.
- **Bin probabilities:** convert Kelvin to Fahrenheit; round each member's approximate high to the nearest whole degree, with half degrees rounded up; count members satisfying each contract's strike condition. This rounding is a transparent modeling assumption and should be revisited during calibration.
- **Outcome:** the contract's resolved YES/NO result. The high temperature is Kalshi's `expiration_value`, which is checked against the bin/result for consistency. Original NWS reports are not independently downloaded, so this column is named `kalshi_settlement_high_f` rather than independently observed temperature.

## Important source change

On checking the API on October 24, 2026, the current series and recent contracts named **The Weather Company**. The July 2026 contracts used here explicitly name **Central Park and the NWS Climatological Report (Daily)**. Do not apply today's series settlement source retroactively to older contracts, or assume this collector applies to newer contracts without reviewing their rules and reporting windows.

## Validation and next work

After collecting, run `python3 inspect_nyc.py` for an independent check of the CSV dimensions, bin partitions, outcomes, forecast availability timestamps, and quote cutoff times. This also creates `data/daily_summary.csv`. It uses only Python's standard library and makes no network requests. Exact packages used for the initial extraction are recorded in `requirements-lock.txt`.

The collector requires six complete contracts per day, 31 complete ensemble members, correct GRIB initialization/lead/variable metadata, dollar-denominated quotes, outcomes consistent with the reported settlement temperature, and bin member counts summing to 31. It aborts on incomplete inputs instead of silently changing ensemble denominators.

Raw empirical probabilities can be zero. Before computing log loss, document a probability-smoothing convention. Fit station bias, spread adjustments, and any smoothing on an earlier training period; evaluate on later dates. Use the same rows for model/market comparisons when quotes are missing. The Brier scores in the validation file are binary-contract averages, not sums over the six-outcome distribution.

Before claiming market-making profitability, obtain an execution model for spreads, fees, fills and inventory. Historical candle midpoints alone cannot establish realizable PnL.

## Sources

- [Kalshi market discovery and Central Park series example](https://docs.kalshi.com/getting_started/quick_start_market_data)
- [Kalshi historical-data routing](https://docs.kalshi.com/getting_started/historical_data)
- [Historical market endpoint](https://docs.kalshi.com/api-reference/historical/get-historical-markets)
- [Historical candlestick endpoint](https://docs.kalshi.com/api-reference/historical/get-historical-market-candlesticks)
- [NOAA GEFS public archive](https://registry.opendata.aws/noaa-gefs/)
- [NWS Central Park station metadata](https://api.weather.gov/stations/KNYC)
- [Kalshi weather settlement and standard-time explanation](https://help.kalshi.com/en/articles/13823837-weather-markets)

Per-contract original rules are preserved in the raw JSON and CSV outputs, and take precedence over general explanatory pages.
