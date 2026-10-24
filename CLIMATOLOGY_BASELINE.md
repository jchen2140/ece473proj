# Week 2: Central Park climatology baseline

The climatology baseline asks: **based only on past seasons, how often would this temperature outcome occur?** It supplies a forecast without looking at the current weather model, market price, or target day's outcome.

## Past temperature records

- **Source:** [NOAA GHCN-Daily Central Park station file](https://www.ncei.noaa.gov/pub/data/ghcn/daily/all/USW00094728.dly).
- **Station:** `USW00094728`, NOAA metadata name `NY CITY CNTRL PARK`, latitude 40.7789, longitude -73.9692. This is the Central Park climate station; its metadata coordinates differ slightly from the KNYC airport-style station listing used for GEFS extraction.
- **Variable:** daily maximum temperature, `TMAX`, stored by NOAA in tenths of degrees Celsius; converted to Fahrenheit with `F = (TMAX / 10) * 9 / 5 + 32`.
- **Quality control:** missing-value code `-9999` and any nonblank quality flag are excluded. Source and measurement flags are retained in the extracted records. [NOAA format and flag definitions](https://www.ncei.noaa.gov/pub/data/ghcn/daily/readme.txt)
- **Download snapshot:** October 24, 2026. The retained history spans January 1, 1991 through October 20, 2026 and contains 13,077 accepted daily highs.
- **Training set:** January 1, 1991–December 31, 2020, containing **10,958 accepted records out of 10,958 expected days: 100% coverage**. This includes leap days. The full archive and metadata can be redownloaded; source manifests retain their URLs and original retrieval timestamps.
- **Post-training set:** 2,119 accepted daily highs from 2021 onward, saved separately. These observations do not affect the fitted baseline and are not all designated as evaluation dates.

## Model

For target calendar date `d`, select all accepted training records whose month/day is within 15 days of `d`, across the 30 training years. This seasonal window pools nearby summer days with summer days and winter days with winter days, while providing more samples than one calendar date alone.

Each selected high is rounded to the nearest whole degree Fahrenheit, with half degrees rounded upward, to form an empirical temperature distribution. For a contract outcome `A`:

```text
P(A | target date) = number of historical sample highs satisfying A
                    / number of accepted historical sample highs
```

For the July 20, 2026 target, the sample contains 930 highs from July 5–August 4 in 1991–2020. Of these, 77 fall in 79–80°F, so the baseline gives that contract **77 / 930 = 8.28%**. The sample's unrounded mean daily high is 85.51°F, and its 10th–90th percentiles are 78.08–93.92°F. These describe historical variability, not a confidence interval for the mean.

Contract-based predictions use the same strict/inclusive strike semantics as the market collector. The interactive CLI's `--lower` and `--upper` bounds are always **inclusive** whole-degree limits. Omit one bound for an open-ended tail.

## Reproduce or query

All baseline commands use Python's standard library. With the committed extracted history, no network access is needed:

```bash
# Create the separated historical datasets and coverage/seasonal summaries
python3 climatology.py

# Forecast a specific outcome on a date after the training period
python3 climatology.py --date 2026-07-20 --lower 79 --upper 80

# Forecast 85°F or above
python3 climatology.py --date 2026-07-20 --lower 85

# Recompute the matched-market baseline and evaluation
python3 milestone1.py

# Verify both the existing pipeline and the new model behavior
python3 -m unittest -v test_milestone1.py test_climatology.py
python3 inspect_nyc.py
```

To reconstruct `historical_tmax.csv` or the uncommitted NOAA raw cache, run `python3 milestone1.py`. It downloads missing source files automatically. Large raw downloads remain excluded from Git.

Python interface:

```python
import datetime as dt
from climatology import SeasonalClimatology, load_history

model = SeasonalClimatology(load_history())
target = dt.date(2026, 7, 20)
model.summary(target)       # Sample size, mean, spread, and quantiles
model.distribution(target)  # Whole-degree temperatures, counts, probabilities
model.cdf(target, 80)       # P(rounded high <= 80°F)
model.probability(target, {
    'strike_type': 'between', 'floor_strike': 79, 'cap_strike': 80,
})
```

## Files to use

| File | Purpose |
| --- | --- |
| `data/historical_tmax.csv` | All accepted extracted station highs from 1991 onward |
| `data/climatology/training_tmax_1991_2020.csv` | Dedicated 30-year baseline training data |
| `data/climatology/post_training_tmax.csv` | Later records, excluded from model fitting |
| `data/climatology/yearly_coverage.csv` | Expected/accepted/missing days and coverage for each training year |
| `data/climatology/monthly_summary.csv` | Monthly sample counts, mean, spread, and quantiles |
| `data/climatology/seasonal_summary.csv` | Baseline summaries for all 366 month/day values, using the seasonal window |
| `data/climatology/manifest.json` | Model settings, source snapshot, missing-date list, and training-file checksum |
| `data/matched_with_climatology.csv` | Baseline probabilities alongside ensemble forecasts and market contracts |
| `data/evaluation.json` | Existing seven-day comparison against ensemble and market forecasts |

## Validation and limitations

The model rejects duplicate training dates, wrong-station records, quality-flagged inputs, inconsistent rounding, empty seasonal windows, and prediction dates inside the fixed training period. Future records are filtered before fitting. Tests cover probability normalization, monotone CDFs, contract boundaries, year wrapping, leap-day handling, and independence from post-training outcomes.

Month/day distances use a circular 366-day reference calendar. Non-leap years have no February 29 observation, so windows near that date have fewer records than a July window. Year-end windows wrap from December to January; only records within the fixed training years are used. Actual accepted sample counts always determine the denominator.

This is a simple empirical baseline, with no climate trend, bias, or spread adjustment. It uses the currently revised NOAA archive rather than archived as-issued data. NOAA GHCN temperatures independently agree, after rounding, with the seven pilot settlement highs, but they do not replace the exact original NWS CLI releases specified by the contracts. The seven-market-day evaluation is a functionality check, not a reliable skill estimate.
