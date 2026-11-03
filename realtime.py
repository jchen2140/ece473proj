"""Which clock does the market run on? Minute-level latency of Kalshi prices to station temperatures.

Run after collect_realtime.py. A knock-out is the minute the station's running maximum (1-minute
ASOS, since local standard midnight) passes a bin's upper edge, for a bin that settled NO. The
1-minute series reads 1F above the official high on many days (rounding between C and F), so the
crossing is measured two ways that bracket the truth:
- first touch: the running maximum first exceeds the edge. Some of these readings round back down
  officially, so a careful market may rightly wait; this overstates the lag.
- certain: the running maximum exceeds the edge by MARGIN_F, when the bin is dead whatever the
  rounding. The market may already have concluded so earlier; this understates the lag.

For each knock-out:
- lag: minutes until the bin's YES price stays at or below DEAD_PRICE for the rest of the day
  (negative: the market had already written the bin off; LOOKBACK caps how early).
- stale trades: contracts traded at YES >= STALE_PRICE after the crossing, and the dollars the YES
  side paid for a contract already worth zero. Taker NO = someone hitting a stale bid (a fast trader);
  taker YES = someone buying a dead contract.
- hourly-report test: minutes until the next hourly METAR showing a temperature above the edge. If
  prices waited for the hourly report, lag would rise one for one with that delay.
Writes data/results/realtime/.
"""
import datetime as dt
import gzip
import math
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import cities
import config
import forecasts
from collect_nyc import DATA, write_csv
from collect_realtime import RAW, REALTIME_CITIES

OUT = DATA / 'results' / 'realtime'
DEAD_PRICE = 0.03
STALE_PRICE = 0.05
MARGIN_F = 1.0
DEFINITIONS = {'first touch': 0.0, 'certain': MARGIN_F}
LOOKBACK = pd.Timedelta(hours=3)  # How early a bin can count as already written off.
RNG = np.random.default_rng(config.BOOTSTRAP_SEED)


def read_gz(path):
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        return pd.read_csv(f)


def load(code):
    city = cities.BY_CODE[code]
    markets = pd.read_csv(DATA / 'cities' / code / 'markets.csv')
    minute = read_gz(RAW / code / 'asos1min.csv.gz')
    minute['time'] = pd.to_datetime(minute['valid_utc'], utc=True)
    minute = minute.sort_values('time')
    trades = read_gz(RAW / code / 'trades.csv.gz')
    trades['time'] = pd.to_datetime(trades['created_time'], utc=True, format='ISO8601')
    trades['yes_price'] = trades['yes_price'].astype(float)
    trades['count'] = trades['count'].astype(float)
    hourly = read_gz(DATA / 'cities' / code / 'asos.csv.gz')
    hourly['time'] = pd.to_datetime(hourly['valid_utc'], utc=True)
    hourly = hourly[hourly['time'].dt.minute >= 45].sort_values('time')  # Routine reports at :51-:56.
    return city, markets, minute, trades.sort_values('time'), hourly


def knockouts(city, markets, minute, trades, hourly):
    events, checks = [], []
    by_ticker = {t: g for t, g in trades.groupby('market_ticker')}
    for day, group in markets.groupby('target_date'):
        start = pd.Timestamp(cities.day_start_utc(city, dt.date.fromisoformat(day)))
        temps = minute[(minute['time'] >= start) & (minute['time'] < start + pd.Timedelta(days=1))]
        if len(temps) < 600:  # Too much of the day missing to trust the running maximum.
            continue
        running = temps['tmpf'].cummax().to_numpy()
        checks.append({'day': day, 'one_minute_max': running[-1], 'settled': group['settlement_high_f'].iloc[0]})
        day_hourly = hourly[(hourly['time'] >= start) & (hourly['time'] < start + pd.Timedelta(days=1))]
        for r in group.to_dict('records'):
            lo, hi = forecasts.bin_edges({k: ('' if pd.isna(v) else v) for k, v in r.items()})
            t = by_ticker.get(r['market_ticker'])
            if math.isinf(hi) or r['outcome'] != 0 or t is None:
                continue
            for definition, margin in DEFINITIONS.items():
                past = running > hi + margin
                if not past.any():
                    continue
                crossing = temps['time'].iloc[int(np.argmax(past))]
                window = t[t['time'] >= crossing - LOOKBACK]
                alive = window[window['yes_price'] > DEAD_PRICE]
                dead_at = alive['time'].iloc[-1] if len(alive) else crossing - LOOKBACK
                after = t[t['time'] >= crossing]
                stale = after[after['yes_price'] >= STALE_PRICE]
                shown = day_hourly[(day_hourly['time'] >= crossing) & (day_hourly['tmpf'] > hi)]
                paid = stale['yes_price'] * stale['count']
                events.append({'city': city.code, 'definition': definition, 'day': day, 'market_ticker': r['market_ticker'],
                               'edge_f': hi, 'crossing_utc': crossing.isoformat(),
                               'local_hour': (crossing - start).total_seconds() / 3600,
                               'lag_min': (dead_at - crossing).total_seconds() / 60,
                               'hourly_report_delay_min': (shown['time'].iloc[0] - crossing).total_seconds() / 60 if len(shown) else np.nan,
                               'trades_after': len(after), 'stale_trades': len(stale), 'stale_contracts': stale['count'].sum(),
                               'stale_dollars': float(paid.sum()),
                               'stale_dollars_taken_by_no_takers': float(paid[stale['taker_side'] == 'no'].sum())})
    return pd.DataFrame(events), pd.DataFrame(checks)


def median_ci(values):
    values = np.asarray(values, float)
    draws = [np.median(RNG.choice(values, len(values))) for _ in range(1000)]
    return float(np.median(values)), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def slope(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    x = x - x.mean()
    return float((x * (y - y.mean())).sum() / (x * x).sum())


def lag_shares(lags):
    lags = np.asarray(lags, float)
    return {'dead_before_crossing': float((lags <= 0).mean()), 'dead_within_5min': float((lags <= 5).mean()),
            'dead_within_15min': float((lags <= 15).mean()), 'dead_within_60min': float((lags <= 60).mean()),
            'dead_after_60min': float((lags > 60).mean())}


def summarize(code, definition, events, checks, contracts_per_day):
    lag, lo, hi = median_ci(events['lag_min']) if len(events) else (np.nan,) * 3
    pending = events.dropna(subset=['hourly_report_delay_min'])
    pending = pending[pending['lag_min'] > 0]  # The report can only matter for bins the market had not written off.
    return {'city': code, 'definition': definition, 'contracts_per_day': contracts_per_day, 'days_with_minute_data': len(checks),
            'one_minute_max_equals_settlement': float((checks['one_minute_max'] == checks['settled']).mean()),
            'within_1f': float(((checks['one_minute_max'] - checks['settled']).abs() <= 1).mean()),
            'knockouts': len(events), 'median_lag_min': lag, 'lag_ci_low': lo, 'lag_ci_high': hi, **lag_shares(events['lag_min']),
            'median_hourly_report_delay_min': float(pending['hourly_report_delay_min'].median()) if len(pending) else np.nan,
            'lag_on_report_delay_slope': slope(pending['hourly_report_delay_min'], pending['lag_min']) if len(pending) > 10 else np.nan,
            'events_with_stale_trades': float((events['stale_trades'] > 0).mean()),
            'stale_dollars_per_event': float(events['stale_dollars'].mean()),
            'stale_dollars_per_day': float(events['stale_dollars'].sum() / max(1, len(checks))),
            'share_of_stale_dollars_to_fast_traders': float(events['stale_dollars_taken_by_no_takers'].sum() / max(1e-9, events['stale_dollars'].sum()))}


def plot_lag_cdf(events, order, path):
    fig, axes = plt.subplots(1, len(DEFINITIONS), figsize=(11, 4.2), sharey=True)
    grid = np.arange(-60, 181)
    for ax, definition in zip(axes, DEFINITIONS):
        sub = events[events['definition'] == definition]
        for code in order:
            values = sub.loc[sub['city'] == code, 'lag_min'].to_numpy()
            ax.plot(grid, [(values <= g).mean() for g in grid], label=f'{code} ({len(values)})')
        ax.axvline(0, color='#555', lw=0.8)
        ax.set_title(f'{definition} (edge + {DEFINITIONS[definition]:.0f}°F)')
        ax.set_xlabel('Minutes after the 1-minute temperature rules the bin out')
    axes[0].set_ylabel(f'Share of bins priced at ≤{DEAD_PRICE * 100:.0f}¢ for good')
    axes[0].legend(fontsize=8, title='City (knock-outs)')
    fig.suptitle('How fast a bin\'s price dies once the station temperature rules it out')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_liquidity(summary, path):
    d = pd.DataFrame([s for s in summary if s['definition'] == 'certain'])
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, (col, label) in zip(axes, (('dead_within_15min', 'Share of bins dead within 15 min'),
                                       ('stale_dollars_per_day', 'Dollars paid for dead contracts per day'))):
        ax.scatter(d['contracts_per_day'], d[col], color='#3b6ea8')
        for _, r in d.iterrows():
            ax.annotate(r['city'], (r['contracts_per_day'], r[col]), fontsize=8, xytext=(3, 2), textcoords='offset points')
        ax.set_xscale('log')
        ax.set_xlabel('Contracts traded per day')
        ax.set_ylabel(label)
    fig.suptitle('Real-time latency against liquidity ("certain" knock-outs)')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    (OUT / 'figures').mkdir(parents=True, exist_ok=True)
    all_events, summary = [], []
    for code in REALTIME_CITIES:
        if not (RAW / code / 'trades.csv.gz').exists():
            continue
        city, markets, minute, trades, hourly = load(code)
        events, checks = knockouts(city, markets, minute, trades, hourly)
        per_day = float(trades['count'].sum() / max(1, trades['time'].dt.date.nunique()))
        for definition in DEFINITIONS:
            summary.append(summarize(code, definition, events[events['definition'] == definition], checks, per_day))
        all_events.append(events)
        print(f'{code}: {len(events)} knock-out measurements', flush=True)
    events = pd.concat(all_events, ignore_index=True)
    order = [s['city'] for s in sorted(summary, key=lambda s: -s['contracts_per_day']) if s['definition'] == 'certain']
    write_csv('results/realtime/knockouts.csv', events.to_dict('records'))
    write_csv('results/realtime/city_summary.csv', summary)
    plot_lag_cdf(events, order, OUT / 'figures' / 'lag_cdf.png')
    plot_liquidity(summary, OUT / 'figures' / 'latency_vs_liquidity.png')

    lines = ['# Real-time latency of Kalshi prices to station temperatures (generated by realtime.py; do not edit by hand)', '',
             f'{len(order)} cities, {cities.WINDOW_START} to {cities.WINDOW_END}. A knock-out is the minute the 1-minute ASOS running '
             'maximum passes a bin\'s upper edge, for a bin that then settled NO. "First touch" uses the edge itself and overstates '
             f'the lag; "certain" requires {MARGIN_F:.0f}°F past the edge, which the official rounding cannot undo, and understates it. '
             f'Lag = minutes until the bin\'s YES price stays at or below {DEAD_PRICE:.2f} for the rest of the day (bins already dead '
             f'for the {LOOKBACK.total_seconds() / 3600:.0f} h before count as {LOOKBACK.total_seconds() / 60:.0f} min early). '
             f'Stale = trades at YES ≥ {STALE_PRICE:.2f} after the crossing; dollars are what the YES side paid for a worthless contract.', '']
    for definition in DEFINITIONS:
        sub = events[events['definition'] == definition]
        p = lag_shares(sub['lag_min'])
        lines.append(f"Pooled, {definition}: {len(sub):,} knock-outs; {p['dead_before_crossing']:.0%} priced dead before the crossing, "
                     f"{p['dead_within_5min']:.0%} within 5 min, {p['dead_within_15min']:.0%} within 15, {p['dead_within_60min']:.0%} "
                     f"within an hour, {p['dead_after_60min']:.0%} later.")
    lines += ['', '| City | Definition | Contracts/day | Knock-outs | Median lag (min) [95% CI] | Dead before | ≤5 min | ≤15 min | >60 min | '
              'Hourly report delay (min) | Lag on report delay | Events with stale trades | $ paid for dead contracts / day | Share to fast traders |',
              '| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for code in order:
        for s in (s for s in summary if s['city'] == code):
            lines.append(f"| {code} | {s['definition']} | {s['contracts_per_day']:,.0f} | {s['knockouts']} | "
                         f"{s['median_lag_min']:.1f} [{s['lag_ci_low']:.1f}, {s['lag_ci_high']:.1f}] | {s['dead_before_crossing']:.0%} | "
                         f"{s['dead_within_5min']:.0%} | {s['dead_within_15min']:.0%} | {s['dead_after_60min']:.0%} | "
                         f"{s['median_hourly_report_delay_min']:.0f} | {s['lag_on_report_delay_slope']:+.2f} | {s['events_with_stale_trades']:.0%} | "
                         f"${s['stale_dollars_per_day']:,.0f} | {s['share_of_stale_dollars_to_fast_traders']:.0%} |")
    lines += ['', '"Lag on report delay": slope of lag on minutes until the hourly report shows the crossing, for bins not yet written off. '
              '1 would mean prices wait for the hourly report; 0 that they ignore it.', '',
              '| City | Days with 1-minute data | 1-minute max = settled high | Within 1°F |', '| --- | ---: | ---: | ---: |']
    lines += [f"| {s['city']} | {s['days_with_minute_data']} | {s['one_minute_max_equals_settlement']:.0%} | {s['within_1f']:.0%} |"
              for s in summary if s['definition'] == 'certain']
    lines += ['', '## Caveats', '',
              '- The 1-minute archive is the sensor record, not a public feed; the public 5-minute and hourly feeds are samples of it. '
              'A lag at or below zero means traders saw the same information through some feed, or extrapolated the warming trend.',
              '- Only bins that settled NO are counted, so a 1-minute reading that the official record rounds away cannot create a '
              'false knock-out.',
              f'- Price death is judged from trades, so a bin that stops trading is dead at its last trade above {DEAD_PRICE:.2f}.']
    text = '\n'.join(lines) + '\n'
    (OUT / 'realtime_tables.md').write_text(text, encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
