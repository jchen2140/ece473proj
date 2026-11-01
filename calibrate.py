"""Calibrated NOAA forecast: a rolling Gaussian correction of the GEFS ensemble.

For each target day, fit on verified days up to D-2 only (the D-1 high is not final at the noon
cutoff). A fixed fit on train days would fail here: the mean error is +1.30F on train days and
-2.11F on test days, so the correction has to move with the season. Candidates are compared on
train days only; the winner is applied to every day and written to data/noaa_calibrated.csv.
"""
import csv
import datetime as dt
import itertools
import json
import math
import statistics

import config
import forecasts
from collect_nyc import DATA, write_csv

WINDOWS = (14, 21, 30, 45, 60, 90)
MEANS = ('bias', 'regression')
SPREADS = ('errors', 'scaled')
MIN_DAYS = 14
EVAL_START = dt.date(2025, 9, 1)  # Same first day for every candidate, after the shortest warm-up.


def history(day, window, moments, highs):
    """(ensemble mean, ensemble variance, high) for verified days in the window, oldest last."""
    target = dt.date.fromisoformat(day)
    rows = []
    for back in range(2, window + 2):
        d = (target - dt.timedelta(days=back)).isoformat()
        if d in moments and d in highs:
            rows.append((*moments[d], highs[d]))
    return rows


def regression_fit(x, y):
    """Least-squares (intercept, slope); a flat window falls back to slope one."""
    mx, my = statistics.fmean(x), statistics.fmean(y)
    denom = sum((v - mx) ** 2 for v in x)
    slope = sum((a - mx) * (b - my) for a, b in zip(x, y)) / denom if denom else 1.0
    return my - slope * mx, slope


def predict(day, moments, highs, window, mean_method, spread_method):
    """Mean and sd for day, or None if too few verified days are available."""
    rows = history(day, window, moments, highs)
    if len(rows) < MIN_DAYS:
        return None
    x = [r[0] for r in rows]
    y = [r[2] for r in rows]
    mean_today, var_today = moments[day]
    if mean_method == 'bias':
        shift = statistics.fmean(b - a for a, b in zip(x, y))
        mu = mean_today + shift
        residuals = [b - a - shift for a, b in zip(x, y)]
    else:
        intercept, slope = regression_fit(x, y)
        mu = intercept + slope * mean_today
        residuals = [b - intercept - slope * a for a, b in zip(x, y)]
    sd = statistics.stdev(residuals)
    if spread_method == 'scaled':
        # Half of the error variance follows today's ensemble spread relative to the window.
        typical = statistics.fmean(r[1] for r in rows)
        sd *= math.sqrt(0.5 + 0.5 * var_today / typical) if typical else 1.0
    return mu, max(forecasts.MIN_SD, sd)


def day_probabilities(group, fit):
    return forecasts.floor_probabilities(forecasts.gaussian_bins(group, *fit))


def losses(p, outcomes):
    brier = sum((a - b) ** 2 for a, b in zip(p, outcomes)) / len(p)
    log_loss = -sum(y * math.log(a) + (1 - y) * math.log(1 - a) for a, y in zip(p, outcomes)) / len(p)
    return brier, log_loss


def candidate_scores(groups, moments, highs):
    """Mean train-day scores for each candidate on identical days."""
    days = [d for d in sorted(groups) if config.split_of(d) == 'train' and dt.date.fromisoformat(d) >= EVAL_START]
    rows = []
    for window, mean_method, spread_method in itertools.product(WINDOWS, MEANS, SPREADS):
        scored = []
        for day in days:
            group = groups[day]
            fit = predict(day, moments, highs, window, mean_method, spread_method)
            scored.append(losses(day_probabilities(group, fit), [int(r['outcome']) for r in group]))
        rows.append({'window': window, 'mean_method': mean_method, 'spread_method': spread_method,
                     'train_days': len(days), 'train_brier': statistics.fmean(s[0] for s in scored),
                     'train_log_loss': statistics.fmean(s[1] for s in scored)})
    return rows


def main():
    groups = forecasts.day_groups()
    moments = forecasts.ensemble_moments()
    highs = {d: float(g[0]['kalshi_settlement_high_f']) for d, g in groups.items()}
    table = candidate_scores(groups, moments, highs)
    write_csv('calibration_candidates.csv', sorted(table, key=lambda r: r['train_log_loss']))
    best = min(table, key=lambda r: r['train_log_loss'])
    settings = (best['window'], best['mean_method'], best['spread_method'])
    rows = []
    for day, group in sorted(groups.items()):
        fit = predict(day, moments, highs, *settings)
        if fit is None:
            continue
        for r, p in zip(group, day_probabilities(group, fit)):
            rows.append({'target_date': day, 'market_ticker': r['market_ticker'], 'noaa_calibrated_probability': p,
                         'forecast_mean_f': fit[0], 'forecast_sd_f': fit[1]})
    write_csv('noaa_calibrated.csv', rows)
    (DATA / 'noaa_calibration.json').write_text(json.dumps({'selected': best, 'days': len(rows) // 6}, indent=2), encoding='utf-8')
    print('Selected on train days:', best)


if __name__ == '__main__':
    main()
