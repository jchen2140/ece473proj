# Milestone 1: first two weeks, NYC Central Park

The project one-pager schedules the first two weeks as October 6–12 and October 13–19. Those are work deadlines, not a requirement that historical observations come from those dates. This pilot uses July 20–26, 2026, when the contracts explicitly referenced Central Park/NWS, and keeps the agreed one-city scope.

## Deliverable coverage

| Week | One-pager task | NYC implementation / evidence |
| --- | --- | --- |
| 1 | Set up GitHub repo; find/pull settled Kalshi markets (Yohana) | Repository: [jchen2140/ece473proj](https://github.com/jchen2140/ece473proj). `collect_nyc.py`, `data/markets.csv`, and original API JSON provide six resolved bins for each of seven days. |
| 1 | Find/pull ensembles for the same cities (Jessie) | `data/ensemble_temperatures.csv`: 31 archived GEFS members per day, matched to Central Park coordinates and target dates; initialization, valid time, grid distance, and archive timestamps retained. |
| 1 | Find/pull past temperatures for climatology (Zara) | `milestone1.py` retrieves NOAA GHCN Central Park station `USW00094728`, checks its metadata, parses daily TMAX and excludes missing/quality-flagged values; `data/historical_tmax.csv`. |
| 2 | Start climatology baseline (Yohana) | Empirical temperature distribution from 1991–2020 observations within +/-15 calendar days of each target date. Contract-bin probabilities and training counts appear in `data/matched_with_climatology.csv` and `data/climatology_training_summary.csv`. |
| 2 | Convert ensemble forecasts into outcome probabilities (Jessie) | `collect_nyc.py` takes each member's approximate daily high, applies documented rounding, and counts members in each contract bin. Probabilities sum to one per event; no member is silently omitted. |
| 2 | Determine evaluation methods; start evaluation code (Zara) | `milestone1.py` compares ensemble, climatology, and market on identical contract rows using Brier score and binary log loss; `data/evaluation.json`. Ten probability bins and their sample counts are provided in `data/calibration.csv`. |

**Jessie's first two weeks are implemented for NYC.** The data and code also cover the other teammates' data, baseline, and evaluation tasks for this city. The eventual multi-city study remains outside this completed pilot.

The expanded [week 2 baseline guide](CLIMATOLOGY_BASELINE.md) covers the reusable model and a dedicated 10,958-record training dataset with complete coverage, separate post-training records, monthly/seasonal summaries, and provenance manifest in `data/climatology/`.

## Verified pilot results

The completed run contains 42 contracts, 2,170 GEFS temperature samples, 217 member-level daily maxima, and 13,077 accepted historical TMAX records from 1991 through available 2026 dates. Each target day has 930 climatology samples (31 calendar days across 30 historical years). The seven independent GHCN rounded daily highs agree with the seven Kalshi-reported settlement highs.

| Forecast | Mean binary Brier score | Mean binary log loss |
| --- | --- | --- |
| GEFS ensemble | 0.143254 | 0.430066 |
| Seasonal climatology | 0.168702 | 0.538986 |
| Market midpoint | 0.120426 | 0.367695 |

These descriptive results use the same 42 contracts from seven events. They demonstrate the evaluator works and are not evidence of a stable model ranking. Four unit tests cover GHCN quality/missing-value handling, leap/year boundaries, historical sample selection, contract-bin boundaries, and zero-probability scoring.

## Run the complete workflow

```bash
# First-time environment setup
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt

# Requires network unless data are already cached
.venv/bin/python collect_nyc.py
python3 milestone1.py
python3 climatology.py

# Offline verification
python3 inspect_nyc.py
python3 -m unittest -v test_milestone1.py test_climatology.py
```

`collect_nyc.py` decodes GRIB with ecCodes; the baseline, evaluation, and offline checks otherwise use the Python standard library. The full workflow is reproducible from cached source files. Historical source data can be revised on future downloads.

## Evaluation specification

- Outcome: resolved binary contract result. Settlement highs remain labeled as Kalshi-reported values.
- Shared comparison rows: contracts with market midpoints; do not score models on different subsets.
- Brier score: mean squared probability error per binary contract, lower is better.
- Log loss: binary negative log likelihood, natural logarithm, probabilities clipped to `[1e-6, 1-1e-6]` for this metric only. No fitting or smoothing is presented as calibrated probability.
- Calibration: ten fixed bins, with mean prediction, realized YES frequency and count. An empty bin is missing, not zero observed frequency. Seven days are too few for a reliable calibration assessment.
- Historical observations: current NOAA GHCN daily archive provides baseline training data and an independent temperature cross-check. It does not establish which exact NWS CLI release Kalshi used. Differences are reported in `data/observation_crosscheck.csv`, not overwritten.
- Climatology training: 1991–2020 only, +/-15 calendar days, month/day aligned with a leap-year reference calendar; no 2026 evaluation outcomes enter training. Current archive revisions mean the baseline is not strictly vintage locked.

## Scope after these two weeks

Later work includes expanding the evaluation period and cities, training bias/spread adjustments on a separate period, plotting reliable comparisons, and building the LMSR/trader simulation. Those later tasks are not necessary to claim that the first two weeks' NYC forecast data and probability conversion are implemented. Seven test events validate functionality; they do not validate forecast skill or profitability.

Source plan: the supplied `ECE473 Group 14 Project One-Pager.pdf`, Milestone 1 schedule. NOAA source format: https://www.ncei.noaa.gov/pub/data/ghcn/daily/readme.txt . Actual download URLs and retrieval timestamps are retained alongside raw files.
