# Domain-specialised market making for Kalshi weather markets

ECE 473 project, track 1a. We build a market maker specialised for Kalshi's daily high-temperature contracts and test it on real order flow against general-purpose and forecast-only baselines.

**Thesis.** A weather market maker cannot win by out-forecasting the crowd: public forecasts are already in the price before they are published. Its domain advantage comes from building in how weather information *arrives and settles*. The daily high is a running maximum of one named station, so live observations rule outcomes out permanently. A maker that reprices on the live station maximum improves test-day PnL in all seven cities tested over a full year, with a bounded extra worst-case loss; generic levers (confidence scaling, throttling, halting at noon) do not.

**Read [docs/paper.md](docs/paper.md)** for the full write-up: the domain, data and backtest design, forecasting results, how the market behaves, the market-maker ladder, the loss bound and limitations.

## Headline results

| Question | Answer | Source |
|---|---|---|
| Does a domain forecast beat generic ones? | Yes: Brier 0.113 vs 0.123 (calibrated NOAA), 0.145 (raw GEFS), 0.169 (climatology) | `data/results/ml/ml_tables.md` |
| Does it beat the market? | No: Kalshi 0.109; the forecast adds a little in a blend | same |
| Is public forecast information priced in? | Mostly before release; Kalshi removes 27% of the public forecast's error at the noon cutoff, in all 15 cities | `data/results/efficiency/efficiency_tables.md` |
| How fast are station observations priced? | Mostly within minutes; 65–91% of ruled-out bins dead within 15 minutes | `data/results/realtime/realtime_tables.md` |
| What makes a better market maker? | A weather-forecast prior and live running-maximum truncation: each positive in 6/6 cities, pooled significant; truncation's extra tail risk stays within a proven bound | `data/results/adaptive_mm/adaptive_mm_tables.md` |
| What doesn't? | Confidence-scaled liquidity, throttling around information events, halting at noon | same |
| Does it hold over a full year? | Yes: 7 cities, Oct 2025 – Aug 2026 walk-forward (1,990 test city-days); forecast prior positive in 6/7 cities, truncation 7/7 | `data/results/full_year/full_year_tables.md` |
| Where does truncation's tail risk come from? | Lone 1-minute sensor spikes; a pre-registered 5-minute median filter keeps the gain (6/6 held-out cities) with the forecast-only maker's worst days | same, Section 5 |
| What does liquidity cost? | Expected capital = b × the prior's log loss; at equal capital the domain makers earn more; each market has a capacity (best b 30–300) | `data/results/risk_frontier/risk_frontier_tables.md` |
| Does liquidity make prices smarter? | No: it tightens spreads only | `data/results/efficiency/efficiency_tables.md` |

## Layout

```
config.py, cities.py      study windows and train/test splits; the 15-city registry
common.py                 paths, time zones, file helpers
run_pipeline.py           runs every stage in order (python run_pipeline.py --list)
collect/                  downloads: Kalshi, GEFS, ECMWF, NWS MOS, ASOS (hourly and 1-minute)
forecast/                 climatology, persistence, calibrated NOAA, ML model, public forecast
market/                   LMSR, replay backtests, the domain market-maker ladder, synthetic traders
analysis/                 forecast scoring, trader behaviour, information efficiency, latency
tests/                    unit and leakage tests
docs/                     paper.md, data notes, per-milestone pipeline notes
data/                     collected data (raw caches gitignored) and data/results/ outputs
```

## Run

```bash
python -m pip install -r requirements.txt
python run_pipeline.py                 # every analysis stage from the data on disk
python run_pipeline.py --collect       # re-download everything first (hours; resumable)
python run_pipeline.py --only market.makers
python -m unittest discover -s tests -t .
```

Each stage is also a module you can run by hand, for example `python -m analysis.efficiency`. No API keys are needed. Large raw caches (`data/raw/`) are gitignored and rebuilt by `--collect`.

## More detail

- [docs/CROSS_CITY_REPLICATION.md](docs/CROSS_CITY_REPLICATION.md): a frozen-rule Chicago replication of the conservative-truncation procedure, with paired average-PnL and worst-tail uncertainty. Run `python -m market.cross_city` after collecting Chicago's real-time cache and producing the NYC pilot outputs.
- [docs/CONSERVATIVE_TRUNCATION.md](docs/CONSERVATIVE_TRUNCATION.md): a training-only sensor-error margin, current/guarded/no-truncation comparison, coverage audit and NYC risk–return results. Run `python -m market.conservative --cities NYC` after rebuilding that city's real-time cache. Outputs are separate from the existing adaptive-maker results.

- [docs/FULL_YEAR_STUDY.md](docs/FULL_YEAR_STUDY.md): the full-year, seven-city walk-forward ladder, the spike filter and the capital analysis. Run `python -m collect.city_markets --full-year`, `python -m collect.realtime --full-year`, then `python -m market.full_year` and `python -m market.risk_frontier`.
- [docs/CALIBRATION_EXPERIMENT.md](docs/CALIBRATION_EXPERIMENT.md): which forecast correction (bias or spread) matters for scores and for market making at equal risk.
- [docs/MILESTONE2_REPORT.md](docs/MILESTONE2_REPORT.md): Milestone 2 progress report.
- [docs/paper.md](docs/paper.md): the paper.
- [docs/DATA_NOTES.md](docs/DATA_NOTES.md): data decisions, the settlement-source change, model choice.
- [docs/PIPELINE_NOTES.md](docs/PIPELINE_NOTES.md): per-milestone collection and matching detail.
- [docs/CLIMATOLOGY_BASELINE.md](docs/CLIMATOLOGY_BASELINE.md), [docs/MILESTONE1_COVERAGE.md](docs/MILESTONE1_COVERAGE.md): early milestone documents.
