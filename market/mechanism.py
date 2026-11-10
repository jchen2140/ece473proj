"""Why the market maker is an LMSR and not a posted forecast, and what live station data is worth.

Three studies on the six real-time cities, all on the replay in market/makers.py (imported, not changed:
makers.py and conservative.py are hash-locked by the cross-city protocol).

1. Trust curve. If the forecast were the true probability, the best maker would post it and never move
   (b -> infinity). We replay the forecast-prior maker over a range of b, including a fixed-price quoter,
   at each city's train-day fee. Lower b = trust the order flow more; higher b = trust the forecast more.
2. Prior forgetting. Three makers that differ only in the opening price (uniform, public forecast, Kalshi
   noon midpoint) are replayed side by side, and their prices are compared through the day. If the LMSR
   aggregates traders' information, the three should converge regardless of where they started.
3. Value of a minute. The live-truncation gain over the forecast maker as a function of how stale the
   station feed is (0 to 240 minutes).

Settings come from data/results/adaptive_mm/test_summary.csv (the forecast maker's train-day b and fee per
city), so every number here is on the same footing as the paper's same-settings table. Test days only
unless a column says train. A counterfactual replay, as in market/makers.py. Writes data/results/mechanism/.
"""
import csv
import math
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import DATA, write_csv
from market import makers
from market import metrics as mm_metrics

OUT = DATA / 'results' / 'mechanism'
FIXED_B = 1e8  # Effectively infinite liquidity: the price never moves, every crossing order fills in full.
B_CURVE = (10, 30, 100, 300, 1000, 3000, 10000, FIXED_B)
SNAPSHOT_HOURS = (-12, -6, 0, 6, 9, 12, 15, 18, 24)  # Relative to the target day's local-standard midnight.
DELAYS_MIN = (0, 1, 2, 5, 10, 15, 30, 60, 120, 240)
PRIORS = ('uniform', 'public', 'kalshi')


def b_label(b):
    return 'fixed price' if b >= FIXED_B else f'{b:g}'


def city_settings():
    with (DATA / 'results' / 'adaptive_mm' / 'test_summary.csv').open(newline='', encoding='utf-8') as f:
        return {r['city']: (float(r['b0']), float(r['fee'])) for r in csv.DictReader(f) if r['maker'] == 'forecast'}


# ---------------------------------------------------------------- price paths

def price_path(day, prior, b, fee, times):
    """Prices of a plain LMSR (no live data) at each time in `times`, using the fill rule of makers.replay."""
    w = np.asarray(prior, dtype=float).copy()
    out, k = [], 0
    times = sorted(times)
    for ts, outcome, is_yes, limit, size in day['flow']:
        while k < len(times) and ts > times[k]:
            out.append(w / w.sum())
            k += 1
        effective = limit - fee
        if effective <= 0:
            continue
        total = w.sum()
        price = w[outcome] / total
        target = effective if is_yes else 1 - effective
        if (is_yes and price >= target) or (not is_yes and price <= target):
            continue
        reach = abs(b * (math.log(target / (1 - target)) - math.log(price / (1 - price))))
        shares = min(size, reach)
        w[outcome] *= math.exp(shares / b if is_yes else -shares / b)
    while k < len(times):
        out.append(w / w.sum())
        k += 1
    return out


def forgetting(days, b, fee, code):
    rows = []
    for d in days:
        if d['split'] != 'test':
            continue
        times = [d['start'] + h * 3600 for h in SNAPSHOT_HOURS]
        paths = {p: price_path(d, d['priors'][p], b, fee, times) for p in PRIORS}
        for i, h in enumerate(SNAPSHOT_HOURS):
            snap = {p: paths[p][i] for p in PRIORS}
            rows.append({'city': code, 'day': d['day'], 'hour': h,
                         'tv_public_vs_kalshi': 0.5 * float(np.abs(snap['public'] - snap['kalshi']).sum()),
                         'tv_uniform_vs_kalshi': 0.5 * float(np.abs(snap['uniform'] - snap['kalshi']).sum()),
                         **{f'brier_{p}': mm_metrics.brier(snap[p], d['winner']) for p in PRIORS}})
    return rows


# ---------------------------------------------------------------- trust curve

def trust_curve(days, fee, sd_ref, code):
    rows = []
    for b in B_CURVE:
        for maker in ('forecast', 'forecast_live'):
            for split in ('train', 'test'):
                res = [makers.replay(d, maker, b, fee, 1.0, sd_ref) for d in days if d['split'] == split]
                pnl = np.array([r['pnl'] for r in res])
                rows.append({'city': code, 'maker': maker, 'b': b, 'b_label': b_label(b), 'fee': fee, 'split': split,
                             'days': len(pnl), 'mean_pnl': float(pnl.mean()), 'worst_day': float(pnl.min()),
                             'p05_pnl': float(np.percentile(pnl, 5)),
                             'share_profitable': float(np.mean(pnl > 0)),
                             'mean_trading_pnl': float(np.mean([r['trading_pnl'] for r in res])),
                             'mean_fee_income': float(np.mean([r['fee_income'] for r in res])),
                             'filled_shares': float(np.mean([r['filled_shares'] for r in res])),
                             'closing_brier': float(np.mean([r['closing_brier'] for r in res]))})
    return rows


# ---------------------------------------------------------------- value of a minute

def delayed_tilt(delay_s):
    def tilt(day, ts):
        weights = np.ones(len(day['highs']))
        i = np.searchsorted(day['minute_ts'], ts - delay_s, side='right')
        if i == 0:
            return weights, None
        m = day['running_max'][i - 1]
        finite = np.isfinite(day['highs'])
        weights[finite & (day['highs'] < m)] = makers.TOUCHED_WEIGHT
        weights[finite & (day['highs'] < m - 1)] = makers.DEAD_WEIGHT
        return weights, m
    return tilt


def minute_value(days, b, fee, sd_ref, code):
    test = [d for d in days if d['split'] == 'test']
    base = {d['day']: makers.replay(d, 'forecast', b, fee, 1.0, sd_ref)['pnl'] for d in test}
    rows = []
    for delay in DELAYS_MIN:
        fn = delayed_tilt(delay * 60)
        for d in test:
            pnl = makers.replay(d, 'forecast_live', b, fee, 1.0, sd_ref, tilt_function=fn)['pnl']
            rows.append({'city': code, 'delay_min': delay, 'day': d['day'], 'gain': pnl - base[d['day']]})
    return rows


# ---------------------------------------------------------------- report

def money(v):
    return f'-${abs(v):,.0f}' if v < 0 else f'${v:,.0f}'


def plot_trust(curve, path, settings=None):
    d = pd.DataFrame(curve)
    d = d[(d['split'] == 'test') & (d['maker'] == 'forecast')]
    codes = list(dict.fromkeys(d['city']))
    colors = ['#2a6f97', '#c05a2b', '#2a9d8f', '#8a5a9e', '#b08900', '#5c6b73']
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    xs = np.arange(len(B_CURVE))
    for color, code in zip(colors, codes):
        g = d[d['city'] == code].set_index('b').loc[list(B_CURVE)]
        ax.plot(xs, g['mean_pnl'], 'o-', color=color, lw=1.5, ms=4, label=code)
        if settings and code in settings:  # The b each city chose on train days.
            k = list(B_CURVE).index(settings[code][0])
            ax.plot(xs[k], g['mean_pnl'].iloc[k], 'o', ms=10, mfc='none', mec=color, mew=1.5)
    ax.set_yscale('symlog', linthresh=100)
    ax.axhline(0, color='#555', lw=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels([b_label(b).replace('fixed price', 'fixed\nprice') for b in B_CURVE], fontsize=8)
    ax.set_xlabel('LMSR liquidity b: trust the order flow (left) or the forecast (right)')
    ax.set_ylabel('Test PnL / day ($, symlog)')
    ax.legend(fontsize=8, frameon=False, ncol=3, loc='lower left')
    ax.set_title('Posting the forecast and never moving loses money in every city (circles: train-day choice)', fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_forgetting(summary, path):
    d = pd.DataFrame(summary)
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(d['hour'], d['tv_uniform_vs_kalshi'], 'o-', color='#9aa5b1', label='uniform-prior maker vs Kalshi-prior maker')
    ax.plot(d['hour'], d['tv_public_vs_kalshi'], 'o-', color='#2a6f97', label='forecast-prior maker vs Kalshi-prior maker')
    ax.set_xticks(list(SNAPSHOT_HOURS))
    ax.set_xticklabels(['noon\nD-1', '6pm', 'mid-\nnight', '6am', '9am', 'noon', '3pm', '6pm', 'end'], fontsize=8)
    ax.set_ylabel('Price gap (total variation)')
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8, frameon=False)
    ax.set_title('Order flow erases the opening price: six cities, test days, same b and fee', fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_minutes(summary, path):
    d = pd.DataFrame(summary)
    fig, ax = plt.subplots(figsize=(7, 3.8))
    x = np.arange(len(d))
    ax.errorbar(x, d['pooled'], yerr=[d['pooled'] - d['ci_low'], d['ci_high'] - d['pooled']], fmt='o-', color='#2a9d8f',
                ecolor='#555', capsize=3)
    ax.axhline(0, color='#555', lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{m}' for m in d['delay_min']])
    ax.set_xlabel('Station feed delay (minutes)')
    ax.set_ylabel('Truncation gain (sd units, 95% CI)')
    ax.set_title('What a minute of station data is worth: live truncation gain vs feed delay', fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    (OUT / 'figures').mkdir(parents=True, exist_ok=True)
    settings = city_settings()
    curve, forget, minutes = [], [], []
    for code in makers.REALTIME_CITIES:
        if code not in settings or not (makers.RAW / code / 'trades.csv.gz').exists():
            print(f'  {code}: missing settings or real-time cache, skipped', flush=True)
            continue
        b, fee = settings[code]
        days = makers.day_inputs(code)
        sd_ref = float(np.median([d['sd'] for d in days if d['split'] == 'train']))
        forget += forgetting(days, b, fee, code)
        curve += trust_curve(days, fee, sd_ref, code)
        minutes += minute_value(days, b, fee, sd_ref, code)
        print(f'  {code} done (b={b:g}, fee={fee:.2f})', flush=True)
    codes = list(dict.fromkeys(r['city'] for r in curve))

    # Forgetting: mean over city-days, each city weighted equally.
    f = pd.DataFrame(forget)
    forget_summary = (f.groupby(['hour', 'city']).mean(numeric_only=True).groupby('hour').mean().reset_index())
    # Value of a minute: scale each city's daily gains by the sd of its 5-minute gains, then pool.
    m = pd.DataFrame(minutes)
    scale = m[m['delay_min'] == makers.FEED_DELAY // 60].groupby('city')['gain'].std().replace(0, 1.0)
    minute_summary = []
    for delay, g in m.groupby('delay_min'):
        pooled = mm_metrics.bootstrap_mean((g['gain'] / g['city'].map(scale)).to_numpy())
        per_city = g.groupby('city')['gain'].mean()
        minute_summary.append({'delay_min': delay, **{f'{c}_mean': per_city[c] for c in codes},
                               'cities_positive': int((per_city > 0).sum()), 'pooled': pooled['mean'],
                               'ci_low': pooled['ci_low'], 'ci_high': pooled['ci_high']})

    write_csv('results/mechanism/trust_curve.csv', curve)
    write_csv('results/mechanism/prior_forgetting_daily.csv', forget)
    write_csv('results/mechanism/prior_forgetting.csv', forget_summary.to_dict('records'))
    write_csv('results/mechanism/minute_value_daily.csv', minutes)
    write_csv('results/mechanism/minute_value.csv', minute_summary)
    plot_trust(curve, OUT / 'figures' / 'trust_curve.png', settings)
    plot_forgetting(forget_summary.to_dict('records'), OUT / 'figures' / 'prior_forgetting.png')
    plot_minutes(minute_summary, OUT / 'figures' / 'minute_value.png')

    c = pd.DataFrame(curve)
    lines = ['# Mechanism studies (generated by market/mechanism.py; do not edit by hand)', '',
             'Six-city replay of real Kalshi taker flow, test days Jun 1 - Aug 13, 2026. b and fee are each city\'s '
             'train-day choice for the forecast maker (data/results/adaptive_mm/test_summary.csv). Counterfactual estimate.', '',
             '## 1. Trust curve: should the maker just post its forecast?', '',
             'Forecast-prior maker, no live data, at the city\'s train-day fee. "fixed price" is b = 1e8: the forecast is posted '
             'and never moves. Mean test PnL per day (worst day in brackets).', '',
             '| b | ' + ' | '.join(codes) + ' |', '| --- |' + ' ---: |' * len(codes)]
    for b in B_CURVE:
        cells = []
        for code in codes:
            r = c[(c['city'] == code) & (c['b'] == b) & (c['maker'] == 'forecast') & (c['split'] == 'test')].iloc[0]
            cells.append(f"{money(r['mean_pnl'])} ({money(r['worst_day'])})")
        lines.append(f'| {b_label(b)} | ' + ' | '.join(cells) + ' |')
    lines += ['', 'Same, with live running-maximum truncation (forecast_live):', '',
              '| b | ' + ' | '.join(codes) + ' |', '| --- |' + ' ---: |' * len(codes)]
    for b in B_CURVE:
        cells = []
        for code in codes:
            r = c[(c['city'] == code) & (c['b'] == b) & (c['maker'] == 'forecast_live') & (c['split'] == 'test')].iloc[0]
            cells.append(f"{money(r['mean_pnl'])} ({money(r['worst_day'])})")
        lines.append(f'| {b_label(b)} | ' + ' | '.join(cells) + ' |')
    lines += ['', '## 2. Prior forgetting: does the order flow overwrite the opening price?', '',
              'Mean total-variation distance between makers that differ only in their opening prices (0 = identical prices, '
              '1 = disjoint), and each maker\'s Brier score at that time. Cities weighted equally.', '',
              '| Hours from target-day midnight | Forecast vs Kalshi prior | Uniform vs Kalshi prior | Brier uniform | Brier forecast | Brier Kalshi |',
              '| ---: | ---: | ---: | ---: | ---: | ---: |']
    for _, r in forget_summary.iterrows():
        lines.append(f"| {r['hour']:+.0f} | {r['tv_public_vs_kalshi']:.3f} | {r['tv_uniform_vs_kalshi']:.3f} | "
                     f"{r['brier_uniform']:.4f} | {r['brier_public']:.4f} | {r['brier_kalshi']:.4f} |")
    lines += ['', '## 3. Value of a minute: live-truncation gain vs station-feed delay', '',
              'Mean test-day PnL of forecast_live minus forecast, same b and fee. Pooled = each city\'s gains divided by the '
              f'sd of its {makers.FEED_DELAY // 60}-minute gains, all city-days resampled.', '',
              '| Delay (min) | ' + ' | '.join(codes) + ' | Cities positive | Pooled [95% CI] |',
              '| ---: |' + ' ---: |' * len(codes) + ' ---: | --- |']
    for r in minute_summary:
        lines.append(f"| {r['delay_min']} | " + ' | '.join(money(r[f'{c}_mean']) for c in codes) +
                     f" | {r['cities_positive']}/{len(codes)} | {r['pooled']:+.3f} [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}] |")
    lines += ['', '## Caveats', '',
              '- Replay fills only real trades, at the taker\'s own price. A fixed-price quoter would also attract flow that did '
              'not happen on Kalshi; that flow is missing here.',
              '- Feed delay is applied to the 1-minute archive; the true public feed timing is not observed.',
              '- Settings are the forecast maker\'s train choice; curves are diagnostic, not re-tuned on test days.']
    text = '\n'.join(lines) + '\n'
    (OUT / 'mechanism_tables.md').write_text(text, encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
