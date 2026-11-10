# Presentation: "The Thermometer Only Goes Up"

About 12 minutes and 10 slides. Story arc: we tried to win by forecasting, lost to the crowd, worked out why, and found the edge in the settlement rules instead, including the two times the thermometer lied.
Every number comes from `docs/paper.md`; the figure paths are listed per slide.

---

**1. Hook (30 s)**
"Every day on Kalshi, people pay up to $3,000 for contracts that have already lost. We built a market maker that stops selling them."
Visual: one sentence, big.

**2. The game (1 min)**
Six buckets on today's high at Central Park; one pays $1. We are the house: we always quote a price, and we profit if our prices beat the traders'.
Explain LMSR in one line: "a vending machine that raises the price of whatever people buy; b is how stubborn it is."

**3. Plan A: out-forecast the crowd (1.5 min)**
Forecast ladder: climatology 0.169, raw NOAA 0.145, calibrated NOAA 0.123, our ML model 0.113 ... and then Kalshi 0.109.
Punchline: "We built the best forecast we could, and a crowd of gamblers still beat it."
Figure: a simple bar chart of the table in paper Section 4.2 (there is no ready-made one); `data/results/ml/figures/importance_features.png` for the sea-breeze aside.

**4. Why? The crowd knows first (1 min)**
68% of the price move happens *before* the forecast is even published; Kalshi removes 27% of the public forecast's error in all 15 cities.
Figure: `data/results/efficiency/figures/event_study.png`.

**5. So should the house just post its forecast? (1.5 min) — this answers the TA's question**
No: posting the forecast and never moving loses $3,000–$23,000 a day in every city. Informed traders keep buying whatever the forecast gets wrong.
And the LMSR learns fast: within ~12 hours, a maker that opened at our forecast, at Kalshi's price, or at *uniform* quotes the same prices.
Figures: `data/results/mechanism/figures/trust_curve.png`, `prior_forgetting.png`.
Line: "The forecast is the opening bid. The traders teach the house the rest — and they charge tuition."

**6. The insight: the thermometer is a ratchet (1.5 min)**
The daily high can only go up. Once the station reads 92°F, every bucket below 92 is dead forever. Traders still pay $830–$2,970 a day in NYC for dead buckets.
The house can learn this from the thermometer instead of paying informed traders to teach it.
Visual idea: one real afternoon, temperature line climbing, buckets greying out as they die.

**7. It works everywhere (1 min)**
Forecast prior + live truncation: positive in 6/6 cities, pooled significant. The generic tricks (throttling, confidence scaling, halting at noon) do not help.
Figure: `data/results/adaptive_mm/figures/ladder.png`.

**8. What is one minute worth? (1 min)**
With a live feed, truncation earns about twice as much as with a 5-minute-old feed; after 15–30 minutes the edge is gone, because the crowd has already traded on it.
Figure: `data/results/mechanism/figures/minute_value.png`.
Line: "Domain knowledge is only worth money if you're fast enough to use it."

**9. The villain: when the thermometer lies (2 min)**
We proved the extra risk is bounded — *if* the sensor matches the official record.
It didn't, twice: NYC July 31 (sensor 87°F, official 85°F → −$1,419) and LA May 24 (sensor 120°F on a 68°F day → −$1,259).
Fix: a safety margin learned only from training days. NYC's disaster day becomes −$149; it beats the no-truncation maker in all 4 cities where it is active.
Honest twist: in LA the 120°F glitch sets a 52°F margin and switches the fix off, so the next rule needs a sanity check on the sensor.
Figure: `data/results/conservative_truncation/NYC/comparison.png`.

**10. The lesson (30 s)**
"Don't try to know the weather better than the crowd. Know how the bet gets settled." The same idea carries to sports (game clock), elections (precincts reporting) and crypto (oracle updates).

---

## Q&A prep

- *Counterfactual replay?* Fills only happen at prices real traders actually paid; we can't see trades that would have happened only against our quotes.
- *Why LMSR rather than an order book?* A bounded worst-case loss we can prove (paper Section 6.6), and it is the mechanism in the course references.
- *Is it profitable in thin markets?* No: about zero in Austin, Minneapolis and DC. The domain ideas improve a break-even business there.
- *Next step?* Since Aug 14 the contracts settle on a different source, which gives an untouched out-of-sample test period.
