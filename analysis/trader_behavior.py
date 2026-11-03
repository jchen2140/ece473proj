"""How Kalshi traders behave on the Central Park daily-high market, and how they compare to the ML model.

Run after collect/trades.py and forecast/ml_model.py. Uses every taker trade (data/flow/trades.csv.gz)
and hourly candles (data/flow/candles.csv.gz) for the study window. Writes data/results/traders/.

- Activity and price discovery: volume and absolute midpoint moves by hour, against model release times.
- Accuracy by lead time: Brier score of the Kalshi midpoint from two days out to settlement, with the
  noon-cutoff ML and NOAA forecasts as reference levels.
- Taker returns: by price (favorite-longshot bias), trade size, timing, and agreement with the ML model.
  Returns are per contract, held to settlement, with and without Kalshi's taker fee.
"""
import math
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from common import DATA, write_csv

OUT = DATA / 'results' / 'traders'
FEE_RATE = 0.07
PRICE_BINS = [0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 0.90, 0.95, 1.0]
SIZE_BINS = [0, 1, 10, 100, 1000, math.inf]
SIZE_LABELS = ['1', '2-10', '11-100', '101-1,000', '>1,000']
LEAD_HOURS = list(range(-36, 21, 2))  # Hours from local midnight starting the target day (standard time).
# Approximate public-availability times (UTC) of the forecasts traders are likely to watch.
RELEASES = {'GFS/GEFS 00Z': 3.6, 'ECMWF 00Z': 7.5, 'GFS/GEFS 06Z': 9.6, 'NBM 13Z': 14.3,
            'GFS/GEFS 12Z': 15.6, 'ECMWF 12Z': 19.5, 'GFS/GEFS 18Z': 21.6}


def load():
    contracts = pd.read_csv(DATA / 'matched.csv', usecols=['target_date', 'market_ticker', 'outcome', 'prediction_time_utc',
                                                           'kalshi_settlement_high_f'])
    contracts['cutoff'] = pd.to_datetime(contracts['prediction_time_utc'], utc=True)
    # Climate day starts at local standard midnight, 05 UTC.
    contracts['day_start'] = pd.to_datetime(contracts['target_date']).dt.tz_localize('UTC') + pd.Timedelta(hours=5)
    trades = pd.read_csv(DATA / 'flow' / 'trades.csv.gz', dtype={'is_block_trade': str})
    trades = trades[trades['is_block_trade'] != '1'].merge(contracts, on=['target_date', 'market_ticker'])
    trades['time'] = pd.to_datetime(trades['created_time'], utc=True, format='ISO8601')
    yes = trades['taker_side'] == 'yes'
    trades['price'] = np.where(yes, trades['yes_price'], trades['no_price'])
    trades['win'] = np.where(yes, trades['outcome'], 1 - trades['outcome'])
    trades = trades[(trades['price'] > 0) & (trades['price'] < 1)].copy()
    fee = np.ceil(FEE_RATE * trades['count'] * trades['price'] * (1 - trades['price']) * 100) / 100
    trades['return'] = trades['win'] - trades['price']
    trades['return_after_fee'] = trades['return'] - fee / trades['count']
    trades['notional'] = trades['count'] * trades['price']
    trades['hours_from_cutoff'] = (trades['time'] - trades['cutoff']).dt.total_seconds() / 3600
    trades['hours_from_day_start'] = (trades['time'] - trades['day_start']).dt.total_seconds() / 3600
    trades['split'] = trades['target_date'].map(config.split_of)
    candles = pd.read_csv(DATA / 'flow' / 'candles.csv.gz').dropna(subset=['yes_bid', 'yes_ask'])
    candles = candles.merge(contracts, on=['target_date', 'market_ticker'])
    candles['time'] = pd.to_datetime(candles['end_period_ts'], unit='s', utc=True)
    candles['mid'] = (candles['yes_bid'] + candles['yes_ask']) / 2
    candles['hours_from_day_start'] = (candles['time'] - candles['day_start']).dt.total_seconds() / 3600
    return contracts, trades, candles


def weighted(g, column, weight='count'):
    return float(np.average(g[column], weights=g[weight]))


def return_table(trades, by, labels=None):
    rows = []
    for level, g in trades.groupby(by, observed=True):
        # Resample days, then pool contracts within the draw, matching the contract-weighted point estimate.
        sums = g.assign(weighted=g['return'] * g['count']).groupby('target_date')[['weighted', 'count']].sum().to_numpy()
        draws = np.random.default_rng(config.BOOTSTRAP_SEED).integers(0, len(sums), size=(500, len(sums)))
        pooled = sums[draws].sum(axis=1)
        ratios = pooled[:, 0] / pooled[:, 1]
        rows.append({'group': str(level), 'trades': len(g), 'contracts': float(g['count'].sum()),
                     'share_of_contracts': float(g['count'].sum() / trades['count'].sum()),
                     'avg_price': weighted(g, 'price'), 'win_rate': weighted(g, 'win'),
                     'return_per_contract': weighted(g, 'return'), 'after_fee': weighted(g, 'return_after_fee'),
                     'day_ci_low': float(np.percentile(ratios, 2.5)), 'day_ci_high': float(np.percentile(ratios, 97.5))})
    return rows


def lead_accuracy(candles, contracts, reference):
    """Brier of the latest midpoint at each lead, on days where all six bins have a quote then."""
    rows = []
    candles = candles.sort_values('time')
    for lead in LEAD_HOURS:
        at = candles[candles['hours_from_day_start'] <= lead]
        last = at.groupby('market_ticker').tail(1)
        last = last[last['hours_from_day_start'] > lead - 12]  # Stale quotes do not count.
        complete = last.groupby('target_date').filter(lambda g: len(g) == 6)
        for split in ('train', 'test'):
            s = complete[complete['target_date'].map(config.split_of) == split]
            if len(s):
                rows.append({'lead_hours': lead, 'split': split, 'days': s['target_date'].nunique(),
                             'kalshi_brier': float(((s['mid'] - s['outcome']) ** 2).mean()),
                             **{f'{k}_brier': reference[k][split] for k in reference}})
    return rows


def ml_alignment(trades):
    """Taker returns split by whether the trade agreed with the ML probability at the cutoff."""
    ml = pd.read_csv(DATA / 'ml_forecasts.csv', usecols=['market_ticker', 'ml_probability'])
    t = trades.merge(ml, on='market_ticker')
    t = t[(t['hours_from_cutoff'] >= 0) & (t['hours_from_day_start'] < 12)]  # Cutoff to 7am target day.
    p_side = np.where(t['taker_side'] == 'yes', t['ml_probability'], 1 - t['ml_probability'])
    edge = p_side - t['price']
    t['ml_view'] = pd.cut(edge, [-1, -0.10, -0.03, 0.03, 0.10, 1],
                          labels=['ML strongly against', 'ML against', 'ML neutral', 'ML agrees', 'ML strongly agrees'])
    return return_table(t, 'ml_view')


def activity(trades, candles):
    hour = trades['time'].dt.hour + trades['time'].dt.minute / 60
    trades = trades.assign(utc_hour=hour.astype(int))
    volume = trades.groupby('utc_hour')['count'].sum()
    c = candles.sort_values(['market_ticker', 'time'])
    c['move'] = c.groupby('market_ticker')['mid'].diff().abs()
    c = c[(c['hours_from_day_start'] > -36) & (c['hours_from_day_start'] < 0)]  # Before the target day: forecast-driven.
    moves = c.groupby(c['time'].dt.hour)['move'].mean()
    lead = trades.assign(bucket=(trades['hours_from_day_start'] // 3 * 3).clip(-48, 24)).groupby('bucket')['count'].sum()
    return [{'utc_hour': h, 'contracts': float(volume.get(h, 0)), 'share': float(volume.get(h, 0) / volume.sum()),
             'mean_abs_mid_move_before_target_day': float(moves.get(h, np.nan))} for h in range(24)], lead


def plot_activity(rows, path):
    fig, ax = plt.subplots(figsize=(8, 3.6))
    hours = [r['utc_hour'] for r in rows]
    ax.bar(hours, [r['share'] for r in rows], color='#3b6ea8', alpha=0.75, label='Share of contracts traded')
    ax.set_xlabel('UTC hour')
    ax.set_ylabel('Share of volume')
    ax2 = ax.twinx()
    ax2.plot(hours, [r['mean_abs_mid_move_before_target_day'] for r in rows], color='#c05a2b', marker='o', ms=3,
             label='Mean |hourly midpoint move|, before target day')
    ax2.set_ylabel('Mean absolute move ($)')
    for name, h in RELEASES.items():
        ax.axvline(h, color='#888', ls=':', lw=0.8)
        ax.text(h, ax.get_ylim()[1] * 0.98, name, rotation=90, va='top', ha='right', fontsize=7, color='#555')
    lines = ax.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    ax.legend(lines, [l.get_label() for l in lines], loc='upper left', fontsize=8)
    ax.set_title('When Kalshi traders trade and when prices move')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_lead(rows, path):
    d = pd.DataFrame([r for r in rows if r['split'] == 'test'])
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.plot(d['lead_hours'], d['kalshi_brier'], marker='o', ms=3, color='#444', label='Kalshi midpoint')
    names = {'ml': 'ML random forest', 'noaa_calibrated': 'Calibrated NOAA'}
    for k, color in (('ml', '#c05a2b'), ('noaa_calibrated', '#3b6ea8')):
        ax.axhline(d[f'{k}_brier'].iloc[0], color=color, ls='--', label=f'{names[k]} at noon cutoff')
    ax.axvline(-12, color='#888', lw=0.8)
    ax.text(-12, ax.get_ylim()[1], ' noon cutoff (D-1)', va='top', fontsize=8)
    ax.axvline(0, color='#888', lw=0.8)
    ax.text(0, ax.get_ylim()[1], ' target day starts', va='top', fontsize=8)
    ax.set_xlabel('Hours from start of target day (local standard midnight)')
    ax.set_ylabel('Mean Brier per contract')
    ax.set_title('Kalshi accuracy as the target day approaches (test days)')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_longshot(rows, path):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot([0, 1], [0, 1], color='#999', lw=0.8)
    ax.scatter([r['avg_price'] for r in rows], [r['win_rate'] for r in rows], s=[20 + 200 * r['share_of_contracts'] for r in rows],
               color='#3b6ea8')
    ax.set_xlabel('Price paid by taker')
    ax.set_ylabel('Share of contracts that won')
    ax.set_title('Favorite-longshot check (all takers)')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def table(rows, title, first='Group'):
    lines = [f'### {title}', '', f'| {first} | Trades | Share of contracts | Avg price | Win rate | Return/contract | After fee | Day-bootstrap 95% CI |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |']
    for r in rows:
        lines.append(f"| {r['group']} | {r['trades']:,} | {r['share_of_contracts']:.1%} | {r['avg_price']:.3f} | {r['win_rate']:.3f} | "
                     f"{r['return_per_contract']:+.4f} | {r['after_fee']:+.4f} | [{r['day_ci_low']:+.4f}, {r['day_ci_high']:+.4f}] |")
    return lines + ['']


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    (OUT / 'figures').mkdir(parents=True, exist_ok=True)
    contracts, trades, candles = load()
    trades['price_bucket'] = pd.cut(trades['price'], PRICE_BINS)
    trades['size'] = pd.cut(trades['count'], SIZE_BINS, labels=SIZE_LABELS)
    trades['timing'] = pd.cut(trades['hours_from_day_start'], [-1e9, -12, 0, 12, 14, 17, 1e9],
                              labels=['before noon cutoff (D-1)', 'cutoff to midnight', 'overnight to 7am', '7am-9am',
                                      '9am-noon', 'after noon (high often known)'])
    by_price = return_table(trades, 'price_bucket')
    by_size = return_table(trades, 'size')
    by_side = return_table(trades, 'taker_side')
    by_timing = return_table(trades, 'timing')
    by_ml = ml_alignment(trades)
    hourly, lead_volume = activity(trades, candles)

    scores = pd.read_csv(DATA / 'results' / 'ml' / 'scores.csv')
    pick = lambda m, s: float(scores[(scores['model'] == m) & (scores['comparison'] == s)]['brier'].iloc[0])
    reference = {k: {s: pick(k, s) for s in ('train', 'test')} for k in ('ml', 'noaa_calibrated')}
    lead = lead_accuracy(candles, contracts, reference)
    for name, rows in (('returns_by_price', by_price), ('returns_by_size', by_size), ('returns_by_side', by_side),
                       ('returns_by_timing', by_timing), ('returns_by_ml_agreement', by_ml), ('activity_by_hour', hourly),
                       ('accuracy_by_lead', lead)):
        write_csv(f'results/traders/{name}.csv', rows)
    plot_activity(hourly, OUT / 'figures' / 'activity_by_hour.png')
    plot_lead(lead, OUT / 'figures' / 'accuracy_by_lead.png')
    plot_longshot(by_price, OUT / 'figures' / 'favorite_longshot.png')

    test_lead = [r for r in lead if r['split'] == 'test']
    first = test_lead[0]
    if first['kalshi_brier'] <= first['ml_brier']:
        crossing_text = (f"already better than the noon ML forecast at its earliest complete quote ({first['lead_hours']:+d} hours "
                         f"from the target day start): Brier {first['kalshi_brier']:.4f} vs {first['ml_brier']:.4f}")
    else:
        crossing = next((r['lead_hours'] for r in test_lead if r['kalshi_brier'] <= r['ml_brier']), None)
        crossing_text = f"first matches the noon ML forecast's Brier at {crossing:+d} hours from the target day start"
    total = trades['count'].sum()
    lines = ['# Kalshi trader behaviour (generated by analysis/trader_behavior.py; do not edit by hand)', '',
             f"{len(trades):,} non-block taker trades, {total:,.0f} contracts, {trades['notional'].sum():,.0f} dollars of premium, "
             f"{trades['target_date'].nunique()} days. Returns are per contract held to settlement; the fee is Kalshi's taker fee "
             f"(ceil of {FEE_RATE} × count × P × (1 − P) cents per trade). Intervals resample whole days.", '',
             f"- {lead_volume[lead_volume.index < -12].sum() / total:.0%} of contracts trade before the noon cutoff on D-1, "
             f"{lead_volume[(lead_volume.index >= -12) & (lead_volume.index < 0)].sum() / total:.0%} between the cutoff and the target day, "
             f"{lead_volume[lead_volume.index >= 0].sum() / total:.0%} on the target day itself.",
             f"- On test days the Kalshi midpoint is {crossing_text}.", '']
    lines += table(by_price, 'Returns by price paid (favorite-longshot bias)', 'Price paid')
    lines += table(by_size, 'Returns by trade size (contracts per trade)', 'Size')
    lines += table(by_side, 'Returns by taker side', 'Side')
    lines += table(by_timing, 'Returns by when the trade happened', 'Timing')
    lines += table(by_ml, 'Returns by agreement with the ML forecast (trades from the cutoff to 7am on the target day)', 'ML view')
    lines += ['### Accuracy of the Kalshi midpoint by lead time (test days)', '',
              'Hour 0 is local standard midnight starting the target day; the noon cutoff is hour −12.', '',
              '| Hours from target-day start | Days | Kalshi Brier | ML at cutoff | NOAA at cutoff |', '| ---: | ---: | ---: | ---: | ---: |']
    lines += [f"| {r['lead_hours']:+d} | {r['days']} | {r['kalshi_brier']:.4f} | {r['ml_brier']:.4f} | {r['noaa_calibrated_brier']:.4f} |"
              for r in test_lead]
    text = '\n'.join(lines) + '\n'
    (OUT / 'trader_tables.md').write_text(text, encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
