# Full-year, seven-city walk-forward study (Milestone 2)

The six-city maker ladder (`docs/paper.md` Section 6.3) tested 2½ summer months after two spring months of training. This study asks whether its conclusions survive a full year, more cities and a stricter out-of-sample design. It also measures what each maker costs in capital.

## Design

- **Cities:** New York, Los Angeles, Chicago, Miami, Austin, Denver, Philadelphia (`cities.FULL_YEAR_CITIES`).
- **Window:** Aug 14, 2025 – Aug 13, 2026, the whole period the contracts settled on the NWS report.
- **Data:**
  - 365 days of quotes per city and the NBM/GFS MOS forecasts.
  - Hourly and 1-minute ASOS readings.
  - Every Kalshi taker trade: 10.6M trades outside NYC (LA 2.9M, Chicago 1.8M, Miami 1.65M, Austin 1.3M, Denver 1.1M, Philadelphia 0.9M), plus NYC's existing trade file.
  - Collected with `collect.city_markets --full-year` and `collect.realtime --full-year`. Outputs go to `data/cities_full/` and `data/raw/realtime_full/` (gitignored).
- **Walk-forward:**
  - Each test month, every maker's b and fee are chosen from all earlier days whose high was final at that month's first quote. The cutoff is the first of the month minus 2 days.
  - Testing starts in October, after about 6 weeks of history. A city starts later where the conservative margin needs 20 strict days (Austin starts in March).
  - 1,990 test city-days in total. No setting ever sees a test day.
- **Same-settings comparison:** every maker uses the forecast maker's walk-forward b and fee, so a difference comes from the idea alone.
- **True contract bounds:** the open-ended top and bottom bins keep their true infinite edge. A placeholder edge (floor + 3°F) lets live truncation write off a top bin that then wins. `tests/test_full_year.py` shows this case.
- **Jessie's conservative truncation, with her frozen rule:**
  - The margin is the nearest-rank 99th percentile of positive (1-minute max − official high), rounded up to 0.5°F, from strict days (≥ 80% minute coverage and an official high).
  - It is refit every month from earlier days only.
  - `market/makers.py` and `market/conservative.py` are unchanged, so the hashes in `cross_city_protocol.json` still match.

Run it:

```bash
python -m market.full_year --workers 4      # about 25 minutes
python -m market.risk_frontier
```

`grid_daily.csv.gz` (every day × maker × b × fee, 57 MB) is gitignored. The command above rebuilds it. `python -m market.full_year --report-only` rebuilds the tables from the saved test files.

## Results

Tables: `data/results/full_year/full_year_tables.md`. Figure: `data/results/full_year/figures/ladder_full_year.png`.

**What each idea adds, same settings, 7 cities × about 10 months.** "Pooled" divides each city's daily gaps by that city's standard deviation, then resamples all city-days. It rewards a gain that is steady, not just large.

| Idea | Cities positive | Pooled (sd units) [95% CI] |
|---|---:|---|
| Weather forecast prior (L1 − L0) | 6/7 | +0.16 [+0.11, +0.20] |
| Live running-max truncation (L2 − L1) | 7/7 | +0.11 [+0.06, +0.15] |
| Truncation on the Kalshi prior | 7/7 | +0.11 [+0.06, +0.15] |
| Truncation on spike-filtered readings (L2f − L1) | 7/7 | **+0.40 [+0.36, +0.45]** |
| Conservative truncation (L2c − L1) | 3/7 | +0.11 [+0.06, +0.15] |
| Conservative vs plain truncation | 0/7 | −0.10 [−0.15, −0.06] |
| Halting at noon | 1/7 | −0.16 [−0.21, −0.12] |
| Confidence-scaled b | 2/7 | −0.11 [−0.15, −0.07] |

1. **Both domain ideas hold up over a full year and 7 cities.**
   - The forecast prior helps most in spring and summer (+0.21 and +0.34) and is about zero in fall and winter.
   - Truncation is positive in all 7 cities and in winter, spring and summer.
2. **The generic levers still lose.** Halting at noon is worst in winter (−0.41).
3. **Jessie's frozen margin rule does not carry over to a full year.**
   - With a year of strict days the 99th percentile is set by single bad sensor minutes. Margins:
     - NYC 5–8°F.
     - Miami 4–7°F.
     - Philadelphia up to 17°F.
     - Los Angeles 52°F in June, from one corrupt reading.
   - A margin that large switches truncation off, so conservative truncation earns about what the forecast-only maker earns.
   - It removes the tail losses, but also the gain: it loses to plain truncation in all 7 cities.
4. **Every big truncation loss is a lone 1-minute spike.** In NYC each large loss is a day when one minute read above the official high and the rest of the record did not (for example Jul 31: 1-minute 87°F, official 85°F, 5-minute median 84°F).

## The spike filter (pre-registered, tested on 6 held-out cities)

- **Rule:** truncate on the running maximum of a trailing 5-minute rolling median of the 1-minute readings, instead of the raw readings. The tilts, feed delay and settings selection are unchanged.
- **Window:** 5 minutes, to match the public feed's cadence. It was not tuned.
- **Frozen before testing:** the rule was frozen in `data/results/full_year/spike_filter_protocol.json` on Nov 6, 2026, after looking at NYC only. No 1-minute data for the other six cities was on disk then.
- The protocol says to report the result whatever it is.
- After freezing, only the reporting code changed (the CSV writer, the `--report-only` path and this section). The rule's code did not change.

**Held-out cities (LA, Chicago, Miami, Austin, Denver, Philadelphia), same settings:**

| Comparison | Cities positive | Mean gap $/day [95% CI] | Pooled (sd units) [95% CI] |
|---|---:|---|---|
| Filtered − plain truncation | 2/6 | +$4 [−$1, +$11] | +0.02 [−0.02, +0.07] |
| Filtered − forecast only | 6/6 | +$10 [+$9, +$12] | +0.39 [+0.34, +0.44] |

**Tails (mean of the worst 5% of days / worst day):**

| City | Forecast only | Plain truncation | Spike-filtered truncation |
|---|---:|---:|---:|
| NYC | −$334 / −$875 | −$442 / −$1,419 | −$332 / −$876 |
| LAX | −$668 / −$1,103 | −$738 / −$1,259 | −$665 / −$1,103 |
| MIA | −$411 / −$1,025 | −$932 / −$2,147 | −$406 / −$1,025 |
| DEN | −$152 / −$376 | −$209 / −$547 | −$147 / −$376 |
| PHL | −$150 / −$203 | −$285 / −$617 | −$138 / −$202 |

Chicago and Austin barely differ across the three makers. All tail figures are in `same_settings_tails.csv`.

**Reading the result:**
- **The pre-registered mean comparison against plain truncation is not significant.** The filter wins big in Miami (+$19/day) and Philadelphia (+$8/day) and ties elsewhere.
- **The filter keeps truncation's average gain and removes its tail.**
  - Its worst days are the forecast-only maker's worst days, in every city.
  - Its gain over the forecast is positive in all 6 held-out cities.
  - That gain is about four times as steady as plain truncation's (+0.39 vs +0.09 pooled on the held-out cities).
- This is what the conservative margin was meant to do. It works because it uses domain knowledge about how the sensor fails: one bad minute, not a steady offset.
- **The tail comparison was not a pre-registered test.** It is descriptive, and each tail is a handful of days.

## Risk and capital efficiency of b

Tables: `data/results/risk_frontier/risk_frontier_tables.md`. Figure: `data/results/risk_frontier/figures/frontier.png`.

An LMSR with prior π that loses at most b·ln(1/π_w) when outcome w wins gives two capital measures.

- **Expected capital** = mean of b·ln(1/π_winner) = **b × the prior's multiclass log loss**.
  - A better forecast lowers the subsidy at the same liquidity.
  - Log loss per unit of b:

    | City | Uniform | Forecast | Kalshi |
    |---|---:|---:|---:|
    | NYC | 1.79 | 1.56 | 1.35 |
    | MIA | 1.79 | 1.20 | 1.16 |

- **Worst-case capital** = b·ln(1/min π).
  - A sharp prior has a small minimum probability, so it needs more worst-case capital than a uniform one (about 2.3× at b = 300 in NYC).
  - On worst-case capital the uniform maker has the best or tied-best return in every city.

Findings:
1. **At equal expected capital the domain makers earn more.** At $500/day in NYC:

   | Maker | PnL/day |
   |---|---:|
   | Forecast + truncation | $177 |
   | Kalshi copy | $149 |
   | Forecast | $133 |
   | Uniform | $107 |

   At budgets of $50 and $150 a day, forecast + truncation beats the uniform maker in all 7 cities. The Kalshi copy leads in LA. At $500 the thin cities lose money with every maker.
2. **Each market has a capacity.**
   - PnL rises with b, then collapses once the maker supplies more liquidity than the uninformed flow can pay for.
   - The best b is 300 in NYC and LA, 100 in Chicago and Miami, and 30 in Austin, Denver and Philadelphia.
   - The uniform maker collapses at a lower b in every city. A domain prior raises how much capital can be put to work at a profit.
3. **Jessie's equal-risk replay agrees.** Her NYC experiment fixes the worst-case opening bound at $300 per day. There, bias + spread correction earns $14.75/day vs −$7.04 uncorrected, and the Kalshi midpoint $27.79 (`docs/CALIBRATION_EXPERIMENT.md`). At equal risk the ranking is the same as at equal b: a sharper, better-corrected prior earns more.

## Caveats

- **Counterfactual replay:** real trades fill at the taker's own price when the maker is at least as good. Trades the maker would have attracted are missing.
- **Fees:** a hypothetical $0.01–$0.05 maker fee, not Kalshi's schedule.
- **Not blind:**
  - The maker list and grids were fixed after the six-city summer results.
  - The spike filter was designed after seeing NYC's full year, so NYC is not out-of-sample for it.
- **Thin markets:** in Austin, Denver and Philadelphia every maker earns $5–$30/day. The domain ideas improve a small business rather than create a large one.
- **Placeholder tails in the older ladder:** the six-city ladder in `market/makers.py` reads the open-ended bins with finite placeholder edges. That can write off a winning top bin, which can only bias its truncation gains downward. This study uses the true edges.
