# Milestone 2 progress report

Due Nov 12, 2026. Track 1a, domain-specialised market makers. Team: Jessie Chen, Zara Homme, Yohana Tesfaye.

## Planned vs. done

The one-pager's Milestone 2 plan, and where each item stands:

| Planned | Status | Where |
|---|---|---|
| LMSR pricing, trading, inventory and PnL | Done | `market/lmsr.py`, `market/backtest.py`, `market/metrics.py` (Zara) |
| Initialise LMSR prices from NOAA probabilities | Done, and extended: calibrated NOAA, the ML forecast and the public NBM/GFS MOS forecast all serve as priors | `market/backtest.py`, `market/makers.py`; paper §6.2–6.3 |
| Compare against climatology-initialised and plain (uniform) LMSR | Done for NYC, then 6 cities, then 7 cities over a full year | paper §6.2, §6.3, §6.6 |
| Simulate traders who trade on belief vs. quoted price | Done. We then **replaced the simulated flow with real Kalshi taker trades** as the main test, because real flow is harder to fool | `market/traders.py`, `market/calibrate_traders.py` (Zara) |
| Sweep the liquidity parameter b | Done. Every replay sweeps b ∈ {10, 30, 100, 300, 1000} and the fee. The full-year study chooses both walk-forward, and a capital analysis explains the sweep | `market/full_year.py`, `market/risk_frontier.py` |
| Possibly compare with Polymarket | **Dropped.** Polymarket's NYC market settles on LaGuardia, not Central Park. More Kalshi cities with a full year each answer the same question (does the maker generalise?) on matched contracts | — |

## What each person did

**Zara**
- Built the LMSR backtest, NOAA bias and spread correction, the Kalshi flow data, the multi-source ML forecast and the trader analysis.
- Ran the 15-city information-efficiency study, the minute-level latency study and the six-city domain maker ladder, with its loss-bound proposition.
- Reorganised the code into packages and wrote the paper.

**Jessie**
- **Bias vs. spread correction ablation, with an equal-risk replay** (`forecast/calibration_ablation.py`, `market/ablation_replay.py`, `docs/CALIBRATION_EXPERIMENT.md`). When every forecast gets the same $300 worst-case opening bound, correcting the mean matters more than correcting the spread.
- **Conservative truncation** (`market/conservative.py`). A training-estimated sensor margin before writing off bins. It removed the NYC July 31 loss.
- **A frozen-rule Chicago replication** of that procedure (`market/cross_city.py`).

**Yohana**
- **Integration.**
  - Ported Jessie's ablation to the new package layout; the rerun matches her outputs within 3e-12.
  - Added every new stage to `run_pipeline.py`.
  - Merged all three people's work into one branch.
- **Full-year, seven-city collection.** `collect.city_markets --full-year` and `collect.realtime --full-year` gather a year of quotes, forecasts, hourly and 1-minute station data, and every taker trade for New York, LA, Chicago, Miami, Austin, Denver and Philadelphia. That is about 10.6M trades outside NYC.
- **Walk-forward maker ladder** (`market/full_year.py`):
  - Settings are chosen each month from earlier days only, giving 1,990 test city-days.
  - Uses the true open-ended bin edges.
  - Runs Jessie's conservative truncation with her frozen rule.
- **Spike filter.** A pre-registered fix for truncation's tail risk, tested on six held-out cities.
- **Risk and capital analysis of b** (`market/risk_frontier.py`). Expected capital = b × the prior's log loss; mean PnL at equal capital; each market's capacity.
- **Writing.** Paper §6.6–6.8 and updates to the abstract, limitations and conclusion; `docs/FULL_YEAR_STUDY.md`; this report.

## Main results so far

1. **A better forecast does not beat the crowd.** Kalshi's prices beat every forecast we built (Brier 0.109 vs 0.113), and public forecast information is priced before release in all 15 cities.
2. **Two domain ideas make a better market maker, and they hold over a full year in 7 cities.**
   - A weather-forecast prior: positive in 6/7 cities.
   - Truncating prices with the live station maximum: positive in 7/7 cities.
   - Generic levers (halting at noon, confidence-scaled b, throttling) lose money.
3. **Truncation's tail risk comes from lone 1-minute sensor spikes.**
   - A 5-minute median filter, frozen before the held-out data existed, keeps truncation's gain over the forecast (6/6 held-out cities).
   - It brings the worst days back to the forecast-only maker's. Miami's worst-5% average goes from −$932 to −$406.
   - Its pre-registered average gain over plain truncation is not significant: +$4/day [−$1, +$11].
4. **Capital.**
   - Expected capital is b times the prior's log loss, so a better prior is cheaper to run.
   - At equal expected capital, the domain makers earn more than the uniform maker in all 7 cities ($50 and $150 budgets).
   - Each market has a capacity: the best b ranges from 300 in NYC and LA to 30 in Austin, Denver and Philadelphia.

## Changes from the plan, and why

- **Real order flow instead of only simulated traders.** Simulated traders only test what we assume about them. Replaying real Kalshi trades at their own prices is a conservative, counterfactual test.
- **More cities and a full year instead of Polymarket.** This gives matched settlement rules and 1,990 test city-days instead of 105.
- **Walk-forward instead of one train/test split.** Every month is out-of-sample, and the results cover all four seasons.
- **The frozen sensor-margin rule does not scale to a year of data.** Its 99th percentile becomes a single bad minute (margins of 5–17°F, and once 52°F in LA). We kept the rule unchanged and reported this. We then tested a different fix, the spike filter, under a protocol written before the held-out data existed.

## Remaining work

**Slides, Nov 17:**
- The ladder figure: `data/results/full_year/figures/ladder_full_year.png`.
- The capital frontier: `data/results/risk_frontier/figures/frontier.png`.
- The spike-filter tail table.
- The thesis: domain edge comes from how the outcome settles, not from out-forecasting the crowd.

**Final paper, Dec 15:**
- **Re-run the six-city summer ladder with the true bin edges.** It currently uses placeholder edges, which can only understate truncation.
- **Report each maker's tail with intervals.** Add a block bootstrap on the worst-5% mean.
- **Run a sensitivity check on the 5-minute feed delay.**
- **Decide whether to add Minneapolis and Washington DC** to the full-year study.
