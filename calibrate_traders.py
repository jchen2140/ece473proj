"""Fit synthetic-trader settings to real Kalshi flow on train days only.

Size and arrivals come from taker trades between the noon cutoff and noon on the target day.
Informed traders center on the Kalshi midpoint at the end of that window, a proxy for knowing
where the market is heading, so belief noise is set to the size of a typical half spread and
the edge threshold to the median half spread a real taker pays. Informed share cannot be read
from the data, so backtest.py sweeps it. Simulating every real trade is too slow, so arrivals
are capped and sizes scaled up to keep daily volume. Writes data/trader_calibration.json.
"""
import json
import math
import statistics

import numpy as np

import backtest
import config
import forecasts
from collect_nyc import DATA

MAX_ARRIVALS = 300


def main():
    records = [r for r in forecasts.day_records() if config.split_of(r['day']) == 'train']
    flow = backtest.load_flow(records)
    marks = backtest.load_kalshi_marks(records)
    sizes, counts = [], []
    for r in records:
        stop = backtest.stop_time(r['day'], backtest.MAIN_STOP)
        day = [t[4] for t in flow.get(r['day'], []) if t[0] <= stop]
        sizes += day
        counts.append(len(day))
    log_sizes = np.log(np.array(sizes))
    variance = [float(np.mean(np.array(marks[r['day']]) * (1 - np.array(marks[r['day']])))) for r in records if r['day'] in marks]
    groups = forecasts.day_groups()
    half_spreads = [(float(q['market_ask']) - float(q['market_bid'])) / 2 for r in records for q in groups[r['day']]]
    mean_arrivals = statistics.fmean(counts)
    arrivals = min(MAX_ARRIVALS, round(mean_arrivals))
    median_size = float(math.exp(np.median(log_sizes)))
    result = {'source': 'train days, noon cutoff to noon on the target day', 'days': len(records),
              'real_arrivals_per_day': mean_arrivals, 'real_median_size': median_size,
              'real_mean_size': float(np.mean(sizes)), 'real_shares_per_day': float(np.sum(sizes) / len(records)),
              'arrivals': arrivals, 'size_sigma': float(log_sizes.std()),
              'median_size': median_size * mean_arrivals / arrivals,
              # A Dirichlet belief has variance p(1-p)/(concentration+1).
              'belief_concentration': max(1.0, statistics.fmean(variance) / statistics.fmean(h * h for h in half_spreads) - 1),
              'threshold': statistics.median(half_spreads)}
    (DATA / 'trader_calibration.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
