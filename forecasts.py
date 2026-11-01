"""Opening probabilities (LMSR seeds) for each day, all knowable at the noon cutoff.

noaa_calibrated comes from calibrate.py, which must run first.
"""
import csv
import math
import statistics
from collections import defaultdict

from collect_nyc import DATA

MIN_SD = 1.5
UNIFORM_MIX = 0.02  # Every opening price stays positive, so the LMSR is defined.
SEEDS = ['uniform', 'climatology', 'ensemble_smoothed', 'noaa_calibrated', 'kalshi']
LABEL = {'uniform': 'Uniform', 'climatology': 'Climatology', 'ensemble_smoothed': 'Raw GEFS (smoothed)',
         'noaa_calibrated': 'Calibrated NOAA', 'kalshi': 'Kalshi noon'}


def read_rows(name):
    with (DATA / name).open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def normal_cdf(x, mean, sd):
    return 0.5 * (1 + math.erf((x - mean) / (sd * math.sqrt(2))))


def bin_edges(row):
    """Continuous edges of a contract on the rounded whole-degree high."""
    if row['strike_type'] == 'less':
        return -math.inf, float(row['upper_bound']) - 0.5
    if row['strike_type'] == 'greater':
        return float(row['lower_bound']) + 0.5, math.inf
    return float(row['lower_bound']) - 0.5, float(row['upper_bound']) + 0.5


def gaussian_bins(rows, mean, sd):
    return [normal_cdf(hi, mean, sd) - normal_cdf(lo, mean, sd) for lo, hi in map(bin_edges, rows)]


def floor_probabilities(p):
    total = sum(p)
    return [(1 - UNIFORM_MIX) * v / total + UNIFORM_MIX / len(p) for v in p]


def ensemble_moments():
    """target_date -> (mean, variance) of the 31 members' approximate daily highs."""
    groups = defaultdict(list)
    for r in read_rows('ensemble_daily_maxima.csv'):
        groups[r['target_date']].append(float(r['daily_high_approx_f']))
    return {d: (statistics.fmean(v), statistics.variance(v)) for d, v in groups.items()}


def day_groups():
    """target_date -> that day's contract rows in a fixed order."""
    groups = defaultdict(list)
    for r in read_rows('matched_with_climatology.csv'):
        groups[r['target_date']].append(r)
    return {d: sorted(g, key=lambda r: r['market_ticker']) for d, g in sorted(groups.items())}


def day_records():
    """One record per day with quotes and a calibrated forecast: tickers, seeds, winner, cutoff."""
    calibrated = {r['market_ticker']: float(r['noaa_calibrated_probability']) for r in read_rows('noaa_calibrated.csv')}
    records = []
    for day, group in day_groups().items():
        if any(r['market_midpoint'] == '' for r in group) or any(r['market_ticker'] not in calibrated for r in group):
            continue
        n = len(group)
        seeds = {'uniform': [1 / n] * n,
                 'climatology': floor_probabilities([float(r['climatology_probability']) for r in group]),
                 'ensemble_smoothed': [(int(r['members_in_bin']) + 0.5) / (31 + n / 2) for r in group],
                 'noaa_calibrated': [calibrated[r['market_ticker']] for r in group],
                 'kalshi': floor_probabilities([float(r['market_midpoint']) for r in group])}
        records.append({'day': day, 'tickers': [r['market_ticker'] for r in group],
                        'winner': next(i for i, r in enumerate(group) if r['outcome'] == '1'),
                        'cutoff': group[0]['prediction_time_utc'], 'seeds': seeds})
    return records
