"""An information-adaptive LMSR market maker, replayed against real Kalshi taker flow.

Run after collect_realtime.py (and collect_cities.py). Every maker opens at the noon cutoff on D-1
with the public forecast (bias-corrected NBM + GFS MOS, Gaussian bins, from efficiency.py), and
every real taker trade after the cutoff becomes a limit order at the price that taker paid. It
fills only while the maker is at least as good, as in backtest.py. Makers, each adding one idea:

- fixed_noon: fixed b and fee, quoting halted at noon on the target day (the Milestone 2 design).
- fixed_close: the same, quoting until the market closes.
- confidence: b scaled by forecast confidence, b0 * (typical error sd / today's error sd).
- live: confidence, plus prices tilted by the 1-minute station running maximum: a bin whose upper
  edge the maximum has touched is down-weighted, and one it has cleared by 1F is written off.
  Tilting the LMSR prior moves prices without cash changing hands.
- adaptive: live, plus b cut to kappa * b while the temperature is within 1F below a live bin's
  edge, or within 10 minutes after an hourly report (when the latency study finds stale prices
  being picked off). Weights are held fixed when b changes, so prices stay continuous.

Settings are chosen per maker on TRAIN days and scored on TEST days. A counterfactual: real
traders might have acted differently against these quotes. Writes data/results/adaptive_mm/.
"""
import datetime as dt
import math
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import cities
import efficiency
import mm_metrics
from collect_nyc import DATA, write_csv
from collect_realtime import RAW, REALTIME_CITIES

OUT = DATA / 'results' / 'adaptive_mm'
TRAIN_END = dt.date(2026, 5, 31)  # Train Apr-May, test Jun-Aug 13.
B_GRID = (30, 100, 300, 1000)
FEE_GRID = (0.02, 0.05, 0.08)
KAPPA_GRID = (0.1, 0.3)
MAKERS = ('fixed_noon', 'fixed_close', 'confidence', 'live', 'adaptive')
STEPS = (('fixed_close', 'fixed_noon'), ('confidence', 'fixed_close'), ('live', 'confidence'),
         ('adaptive', 'live'), ('adaptive', 'fixed_noon'))
SAME_KAPPA = 0.3  # Fixed in advance for the same-settings comparison.
TOUCHED_WEIGHT = 0.3  # The 1-minute reading can round down officially, so touching an edge is not final.
DEAD_WEIGHT = 1e-3
EDGE_WINDOW_F = 1.0
REPORT_WINDOW = 600  # Seconds after an hourly report.
INFORMED_WINDOW = 900  # Fills within this many seconds after a knock-out or report count as event-time fills.
FEED_DELAY = 300  # The maker sees the public 5-minute feed, so it acts on the reading from 5 minutes earlier.


# ---------------------------------------------------------------- inputs

def day_inputs(code):
    """Per day: bin edges, winner, public prior and its error sd, trades, 1-minute running maximum, report times."""
    city = cities.BY_CODE[code]
    days, panel = efficiency.load_city(city)
    nbm, _ = efficiency.load_runs(city, 'nbm', days)
    gfs, _ = efficiency.load_runs(city, 'gfs_mos', days)
    frame = efficiency.hourly_frame(city, days, panel, nbm, gfs, efficiency.load_obs(city))
    cutoff_rows = frame[frame['hour'] == efficiency.HOURS[0]].set_index('day')
    trades = efficiency.read_gz(RAW / code / 'trades.csv.gz')
    trades['ts'] = pd.to_datetime(trades['created_time'], utc=True, format='ISO8601').astype('int64') / 1e9
    trades['yes_price'] = trades['yes_price'].astype(float)
    trades['count'] = trades['count'].astype(float)
    minute = efficiency.read_gz(RAW / code / 'asos1min.csv.gz')
    minute['ts'] = pd.to_datetime(minute['valid_utc'], utc=True).astype('int64') / 1e9
    minute = minute.sort_values('ts')
    hourly = efficiency.load_obs(city)
    hourly = hourly[hourly['time'].dt.minute >= 45]
    report_ts = np.sort(hourly['time'].astype('int64').to_numpy() / 1e9)
    by_ticker = {t: g for t, g in trades.groupby('market_ticker')}
    out = []
    for day, info in days.items():
        if day not in cutoff_rows.index:
            continue
        row = cutoff_rows.loc[day]
        _, prior = efficiency.public_brier(row, info)
        start = info['start'].timestamp()
        cutoff = start - 12 * 3600
        flow = []
        for i, ticker in enumerate(info['tickers']):
            t = by_ticker.get(ticker)
            if t is None:
                continue
            t = t[t['ts'] >= cutoff].assign(second=lambda x: x['ts'].round(0))
            # Merge same-second fills at one price into one order: identical for the replay, far fewer steps.
            t = t.groupby(['second', 'taker_side', 'yes_price'], as_index=False)['count'].sum()
            for ts, side, price, size in t[['second', 'taker_side', 'yes_price', 'count']].itertuples(index=False):
                limit = price if side == 'yes' else 1 - price
                if 0 < limit < 1:
                    flow.append((ts, i, side == 'yes', limit, size))
        flow.sort()
        temps = minute[(minute['ts'] >= start) & (minute['ts'] < start + 86400)]
        out.append({'day': day, 'split': 'train' if dt.date.fromisoformat(day) <= TRAIN_END else 'test',
                    'prior': np.asarray(prior), 'sd': float(row['error_sd']) if pd.notna(row['error_sd']) else 2.5,
                    'lows': info['lows'], 'highs': info['highs'], 'winner': info['winner'], 'start': start, 'flow': flow,
                    'minute_ts': temps['ts'].to_numpy(), 'running_max': temps['tmpf'].cummax().to_numpy(),
                    'reports': report_ts[(report_ts >= start) & (report_ts < start + 86400)]})
    return out


# ---------------------------------------------------------------- replay

def tilt(day, ts):
    """Multiplicative weights from the station running maximum at time ts (all ones before the target day)."""
    weights = np.ones(len(day['prior']))
    i = np.searchsorted(day['minute_ts'], ts - FEED_DELAY, side='right')
    if i == 0:
        return weights, None
    m = day['running_max'][i - 1]
    finite = np.isfinite(day['highs'])
    weights[finite & (day['highs'] < m)] = TOUCHED_WEIGHT
    weights[finite & (day['highs'] < m - 1)] = DEAD_WEIGHT
    return weights, m


def risky(day, ts, weights, m):
    """Near a live bin's edge, or just after an hourly report."""
    j = np.searchsorted(day['reports'], ts, side='right')
    if j and ts - day['reports'][j - 1] <= REPORT_WINDOW:
        return True
    if m is None:
        return False
    live = (weights == 1) & np.isfinite(day['highs'])
    return bool(np.any((day['highs'][live] - m >= 0) & (day['highs'][live] - m <= EDGE_WINDOW_F)))


def informed_times(day):
    """Times after which fills count as event-time: 1-minute knock-outs and hourly reports."""
    times = list(day['reports'])
    for hi in day['highs'][np.isfinite(day['highs'])]:
        past = day['running_max'] > hi
        if past.any():
            times.append(day['minute_ts'][int(np.argmax(past))])
    return np.sort(np.asarray(times))


def replay(day, maker, b0, fee, kappa, sd_ref):
    prior = day['prior']
    n = len(prior)
    b_day = b0 * (sd_ref / max(day['sd'], 1.0) if maker in ('confidence', 'live', 'adaptive') else 1.0)
    stop = day['start'] + 12 * 3600 if maker == 'fixed_noon' else math.inf
    w = prior.copy()
    applied = np.ones(n)
    shift = np.zeros(n)
    revenue = fee_income = filled = peak = 0.0
    fills = []
    for ts, outcome, is_yes, limit, size in day['flow']:
        if ts > stop:
            break
        b = b_day
        if maker in ('live', 'adaptive'):
            weights, m = tilt(day, ts)
            if not np.array_equal(weights, applied):
                w *= weights / applied
                applied = weights
            if maker == 'adaptive' and risky(day, ts, weights, m):
                b = kappa * b_day
        total = w.sum()
        effective = limit - fee
        if effective <= 0:
            continue
        price = w[outcome] / total
        target = effective if is_yes else 1 - effective
        if (is_yes and price >= target) or (not is_yes and price <= target):
            continue
        reach = abs(b * (math.log(target / (1 - target)) - math.log(price / (1 - price))))
        shares = min(size, reach)
        step = math.exp(shares / b if is_yes else -shares / b)
        grown = total + w[outcome] * (step - 1)
        paid = b * math.log(grown / total)
        if is_yes:
            shift[outcome] += shares
            owed = shares if outcome == day['winner'] else 0.0
        else:
            paid += shares
            shift[np.arange(n) != outcome] += shares
            owed = shares if outcome != day['winner'] else 0.0
        w[outcome] *= step
        revenue += paid
        fee_income += shares * fee
        filled += shares
        peak = max(peak, shift.max() - revenue - fee_income)
        fills.append((ts, paid + shares * fee - owed))
    state = {'revenue': revenue, 'fee_income': fee_income, 'exposure': shift, 'offered': sum(f[4] for f in day['flow']),
             'filled': filled, 'fills': len(fills), 'peak_worst_loss': peak, 'prices': list(w / w.sum())}
    metrics = mm_metrics.day_metrics(state, day['winner'], prior)
    events = informed_times(day)
    if fills:
        ts = np.array([f[0] for f in fills])
        pnl = np.array([f[1] for f in fills])
        k = np.searchsorted(events, ts, side='right')
        recent = (k > 0) & (ts - events[np.maximum(k - 1, 0)] <= INFORMED_WINDOW)
        metrics['event_time_pnl'] = float(pnl[recent].sum())
        metrics['other_pnl'] = float(pnl[~recent].sum())
    else:
        metrics['event_time_pnl'] = metrics['other_pnl'] = 0.0
    return metrics


def run_city(code):
    days = day_inputs(code)
    sd_ref = float(np.median([d['sd'] for d in days if d['split'] == 'train']))
    results = {}
    for maker in MAKERS:
        kappas = KAPPA_GRID if maker == 'adaptive' else (1.0,)
        for b0 in B_GRID:
            for fee in FEE_GRID:
                for kappa in kappas:
                    results[maker, b0, fee, kappa] = [{'day': d['day'], 'split': d['split'], **replay(d, maker, b0, fee, kappa, sd_ref)}
                                                      for d in days]
        print(f'  {code} {maker} done', flush=True)
    return days, results


def select(results):
    chosen = {}
    for maker in MAKERS:
        options = [(np.mean([d['pnl'] for d in v if d['split'] == 'train']), key) for key, v in results.items() if key[0] == maker]
        chosen[maker] = max(options)[1]
    return chosen


# ---------------------------------------------------------------- report

def money(v):
    return f'-${abs(v):,.0f}' if v < 0 else f'${v:,.0f}'


def plot_pnl(rows, path):
    d = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, d['city'].nunique(), figsize=(4 * d['city'].nunique(), 3.6), squeeze=False)
    for ax, (code, g) in zip(axes[0], d.groupby('city', sort=False)):
        g = g.set_index('maker').loc[list(MAKERS)]
        ax.bar(range(len(g)), g['mean_pnl'], yerr=[g['mean_pnl'] - g['ci_low'], g['ci_high'] - g['mean_pnl']],
               color=['#9aa5b1', '#9aa5b1', '#3b6ea8', '#2a9d8f', '#c05a2b'])
        ax.set_xticks(range(len(g)))
        ax.set_xticklabels(g.index, rotation=35, ha='right', fontsize=8)
        ax.axhline(0, color='#555', lw=0.8)
        ax.set_title(code)
    axes[0][0].set_ylabel('Test-day PnL per day ($)')
    fig.suptitle('Market makers on test days (settings chosen on train days)')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    (OUT / 'figures').mkdir(parents=True, exist_ok=True)
    rows, pairs, grid, same_gaps = [], [], [], []
    for code in REALTIME_CITIES:
        if not (RAW / code / 'trades.csv.gz').exists():
            continue
        days, results = run_city(code)
        chosen = select(results)
        for key, v in results.items():
            for split in ('train', 'test'):
                grid.append({'city': code, 'maker': key[0], 'b0': key[1], 'fee': key[2], 'kappa': key[3], 'split': split,
                             **mm_metrics.summarize([d for d in v if d['split'] == split])})
        test = {m: [d for d in results[chosen[m]] if d['split'] == 'test'] for m in MAKERS}
        for m in MAKERS:
            s = mm_metrics.summarize(test[m])
            ci = mm_metrics.bootstrap_mean([d['pnl'] for d in test[m]])
            rows.append({'city': code, 'maker': m, 'b0': chosen[m][1], 'fee': chosen[m][2], 'kappa': chosen[m][3],
                         'days': s['days'], 'mean_pnl': s['mean_pnl'], 'ci_low': ci['ci_low'], 'ci_high': ci['ci_high'],
                         'worst_day': s['worst_day'], 'profitable_days': s['share_days_profitable'], 'filled_shares': s['filled_shares'],
                         'event_time_pnl': s['event_time_pnl'], 'other_pnl': s['other_pnl'], 'peak_worst_case_loss': s['peak_worst_case_loss']})
        for a, b in STEPS:
            gap = mm_metrics.bootstrap_mean([x['pnl'] - y['pnl'] for x, y in zip(test[a], test[b])])
            pairs.append({'city': code, 'difference': f'{a} - {b}', **gap})
        # Same settings for every maker (fixed_close's train choice, kappa fixed in advance), so only the feature differs.
        _, b0, fee, _ = chosen['fixed_close']
        same = {m: [d for d in results[m, b0, fee, SAME_KAPPA if m == 'adaptive' else 1.0] if d['split'] == 'test'] for m in MAKERS}
        for a, b in STEPS:
            for x, y in zip(same[a], same[b]):
                same_gaps.append({'city': code, 'step': f'{a} - {b}', 'day': x['day'], 'gap': x['pnl'] - y['pnl']})
    write_csv('results/adaptive_mm/test_summary.csv', rows)
    write_csv('results/adaptive_mm/paired_differences.csv', pairs)
    write_csv('results/adaptive_mm/grid.csv', grid)
    gaps = pd.DataFrame(same_gaps)
    write_csv('results/adaptive_mm/same_settings_daily_gaps.csv', same_gaps)
    same_rows = []
    for step, g in gaps.groupby('step', sort=False):
        per_city = g.groupby('city', sort=False)['gap'].mean()
        # Scale each city's gaps by its own spread so busy cities do not dominate the pooled test.
        scaled = g['gap'] / g.groupby('city')['gap'].transform(lambda v: v.std() or 1.0)
        pooled = mm_metrics.bootstrap_mean(scaled.to_numpy())
        same_rows.append({'step': step, 'cities_positive': int((per_city > 0).sum()), 'cities': len(per_city),
                          **{f'{c}_mean': v for c, v in per_city.items()}, 'pooled_scaled_mean': pooled['mean'],
                          'pooled_ci_low': pooled['ci_low'], 'pooled_ci_high': pooled['ci_high'], 'pooled_share_positive': pooled['share_positive']})
    write_csv('results/adaptive_mm/same_settings.csv', same_rows)
    plot_pnl(rows, OUT / 'figures' / 'test_pnl.png')

    lines = ['# Information-adaptive market maker (generated by adaptive_mm.py; do not edit by hand)', '',
             f'Real Kalshi taker trades replayed against LMSR makers opened at the noon cutoff with the public forecast. Train '
             f'{cities.WINDOW_START} to {TRAIN_END}, test to {cities.WINDOW_END}; each maker\'s b0, fee and kappa are chosen on train days. '
             'Daily PnL in dollars for one six-bin event. Event-time PnL comes from fills within 15 minutes after a 1-minute knock-out '
             'or an hourly report. A counterfactual estimate, not realized PnL.', '',
             '| City | Maker | b0 | Fee | kappa | Test PnL/day [95% CI] | Worst day | Profitable days | Shares filled/day | '
             'Event-time PnL/day | Other PnL/day | Peak worst-case loss |',
             '| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in rows:
        lines.append(f"| {r['city']} | {r['maker']} | {r['b0']} | {r['fee']:.2f} | {r['kappa']:.1f} | {money(r['mean_pnl'])} "
                     f"[{money(r['ci_low'])}, {money(r['ci_high'])}] | {money(r['worst_day'])} | {r['profitable_days']:.0%} | "
                     f"{r['filled_shares']:,.0f} | {money(r['event_time_pnl'])} | {money(r['other_pnl'])} | {money(r['peak_worst_case_loss'])} |")
    lines += ['', '| City | Difference in test PnL/day | Mean [95% CI] | Share of replicates above 0 |', '| --- | --- | --- | ---: |']
    lines += [f"| {p['city']} | {p['difference']} | {money(p['mean'])} [{money(p['ci_low'])}, {money(p['ci_high'])}] | {p['share_positive']:.3f} |"
              for p in pairs]
    codes = list(gaps['city'].unique())
    lines += ['', '## Same settings for every maker', '',
              f'Each city uses fixed_close\'s train-day b0 and fee for all makers, and kappa {SAME_KAPPA} for the adaptive one, so a '
              'difference comes from the feature, not from tuning. Pooled: each city\'s daily gaps are divided by their own standard '
              'deviation, then all city-days are resampled.', '',
              '| Step | ' + ' | '.join(codes) + ' | Cities positive | Pooled (sd units) [95% CI] | Share of replicates above 0 |',
              '| --- |' + ' ---: |' * len(codes) + ' ---: | --- | ---: |']
    for r in same_rows:
        lines.append(f"| {r['step']} | " + ' | '.join(money(r[f'{c}_mean']) for c in codes) +
                     f" | {r['cities_positive']}/{r['cities']} | {r['pooled_scaled_mean']:+.3f} [{r['pooled_ci_low']:+.3f}, "
                     f"{r['pooled_ci_high']:+.3f}] | {r['pooled_share_positive']:.3f} |")
    lines += ['', '## Caveats', '',
              '- Replay: a real trade fills only if the maker would have been at least as good, at the taker\'s own price. Trades '
              'the maker would have attracted but Kalshi did not are missing, and traders might have reacted to these quotes.',
              f'- The live and adaptive makers act on the 1-minute reading from {FEED_DELAY // 60} minutes earlier, as a stand-in for '
              'the public 5-minute feed; a trader with a faster feed would do better, a slower one worse.',
              f'- Two months of train days choose {len(B_GRID) * len(FEE_GRID)} to {len(B_GRID) * len(FEE_GRID) * len(KAPPA_GRID)} settings per '
              'maker; test results are the evidence.']
    text = '\n'.join(lines) + '\n'
    (OUT / 'adaptive_mm_tables.md').write_text(text, encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
