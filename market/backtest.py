"""Backtest a Kalshi-style daily-high event against an LMSR opened at the noon cutoff.

Replay: every real Kalshi taker trade after the cutoff becomes a limit order at the price that
trader actually paid. It fills against the LMSR only while the LMSR is at least as good, so a
trader never gets a worse deal than the real one. Trades the LMSR would have attracted but
Kalshi did not are invisible, and real traders might have reacted to LMSR prices, so this is a
counterfactual estimate, not realized PnL.

Halt: the main run stops quoting at noon on the target day, before the daily high is usually
set; this was fixed before any results were seen. Quoting to market close is a sensitivity run.
b and the fee are chosen per seed on train days only, then scored once on test days.

Synthetic: market/traders.py populations at the selected settings, varying the informed share.
Run forecast/noaa_calibration.py and market/calibrate_traders.py first. Writes data/results/backtest/: grid and daily
tables, per-contract prices and inventory, and a fill-by-fill log for test days.
"""
import csv
import datetime as dt
import gzip
import json
import math
from collections import defaultdict
from zoneinfo import ZoneInfo

import numpy as np

import config
from forecast import probabilities as forecasts
from market import metrics as mm_metrics
from market import traders
from common import DATA, write_csv
from market.lmsr import LMSR

OUT = DATA / 'results' / 'backtest'
NY = ZoneInfo('America/New_York')
B_GRID = (10, 30, 100, 300, 1000, 3000, 10000)
FEE_GRID = (0.0, 0.01, 0.02, 0.03, 0.05)
STOPS = ('target_noon', 'market_close')
MAIN_STOP = 'target_noon'
INFORMED_SHARES = (0.0, 0.1, 0.25, 0.5, 0.75)
SYNTHETIC_REPLICATES = 3
SYNTHETIC_FEE = 0.01  # Shared by all seeds, and below noise_slippage so noise traders can fill.
PAIRS = [('noaa_calibrated', 'uniform'), ('noaa_calibrated', 'climatology'),
         ('noaa_calibrated', 'ensemble_smoothed'), ('noaa_calibrated', 'kalshi'),
         ('ml', 'noaa_calibrated'), ('ml', 'kalshi')]


def epoch(text):
    """UTC seconds from an ISO time; sub-second parts are dropped because file order keeps sequence."""
    return dt.datetime.fromisoformat(text[:19]).replace(tzinfo=dt.timezone.utc).timestamp()


def stop_time(day, stop):
    if stop == 'market_close':
        return math.inf
    return dt.datetime.combine(dt.date.fromisoformat(day), dt.time(12), NY).timestamp()


def load_flow(records, path=None):
    """day -> time-ordered tuples (ts, outcome, is_yes, limit, shares), non-block trades after the cutoff."""
    tickers = {t: (r['day'], i) for r in records for i, t in enumerate(r['tickers'])}
    cutoffs = {r['day']: dt.datetime.fromisoformat(r['cutoff']).timestamp() for r in records}
    flow = defaultdict(list)
    with gzip.open(path or DATA / 'flow' / 'trades.csv.gz', 'rt', newline='', encoding='utf-8') as f:
        for t in csv.DictReader(f):
            if t['market_ticker'] not in tickers or t['is_block_trade'] == '1':
                continue
            day, outcome = tickers[t['market_ticker']]
            ts = epoch(t['created_time'])
            if ts < cutoffs[day]:
                continue
            is_yes = t['taker_side'] == 'yes'
            limit = float(t['yes_price' if is_yes else 'no_price'])
            if 0 < limit < 1:
                flow[day].append((t['created_time'], ts, outcome, is_yes, limit, float(t['count'])))
    return {d: [v[1:] for v in sorted(rows, key=lambda v: v[0])] for d, rows in flow.items()}


def load_kalshi_marks(records, path=None):
    """(day, stop) -> latest quote midpoints no later than stop, or absent if any bin has no quote."""
    wanted = {t: (r['day'], i) for r in records for i, t in enumerate(r['tickers'])}
    quotes = defaultdict(dict)
    with gzip.open(path or DATA / 'flow' / 'candles.csv.gz', 'rt', newline='', encoding='utf-8') as f:
        for c in csv.DictReader(f):
            if c['market_ticker'] in wanted and c['yes_bid'] != '' and c['yes_ask'] != '':
                day, outcome = wanted[c['market_ticker']]
                ts = float(c['end_period_ts'])
                if ts <= stop_time(day, MAIN_STOP):
                    best = quotes[day].get(outcome)
                    if best is None or ts > best[0]:
                        quotes[day][outcome] = (ts, (float(c['yes_bid']) + float(c['yes_ask'])) / 2)
    return {d: [q[i][1] for i in range(6)] for d, q in quotes.items() if len(q) == 6}


def replay_day(flow, priors, b, fee, stop_at, log=None):
    """Same arithmetic as lmsr.LMSR with O(1) updates, since the grid replays millions of trades.

    w holds exp(q/b) up to a common factor. NO shares are applied as YES -x plus x guaranteed
    shares, so each trade touches one weight and test_backtest checks this against LMSR.
    """
    n = len(priors)
    w = list(priors)
    total = 1.0
    shift = [0.0] * n   # Shares owed per outcome from this day's fills.
    revenue = fee_income = offered = filled = peak = 0.0
    fills = 0
    for ts, outcome, is_yes, limit, size in flow:
        if ts > stop_at:
            break
        offered += size
        effective = limit - fee
        if effective <= 0:
            continue
        price = w[outcome] / total
        target = effective if is_yes else 1 - effective
        if is_yes:
            if price >= target:
                continue
        elif price <= target:
            continue
        reach = abs(b * (math.log(target / (1 - target)) - math.log(price / (1 - price))))
        shares = min(size, reach)
        step = math.exp(shares / b if is_yes else -shares / b)
        grown = total + w[outcome] * (step - 1)
        paid = b * math.log(grown / total)
        if is_yes:
            shift[outcome] += shares
        else:
            paid += shares
            for i in range(n):
                if i != outcome:
                    shift[i] += shares
        w[outcome] *= step
        total = grown
        revenue += paid
        fee_income += shares * fee
        filled += shares
        fills += 1
        peak = max(peak, max(shift) - revenue - fee_income)
        if log is not None:
            log.append({'ts': ts, 'outcome': outcome, 'side': 'yes' if is_yes else 'no', 'real_price': limit,
                        'shares': shares, 'paid': paid, 'fee': shares * fee, 'lmsr_price_after': w[outcome] / total,
                        'exposure_after': shift[outcome], 'worst_case_loss': max(shift) - revenue - fee_income})
    return {'revenue': revenue, 'fee_income': fee_income, 'exposure': shift, 'offered': offered,
            'filled': filled, 'fills': fills, 'peak_worst_loss': peak, 'prices': [v / total for v in w]}


def run_replay(records, flow, marks):
    results = {}
    for stop in STOPS:
        for seed in forecasts.SEEDS:
            for b in B_GRID:
                for fee in FEE_GRID:
                    days = []
                    for r in records:
                        priors = r['seeds'][seed]
                        state = replay_day(flow.get(r['day'], []), priors, b, fee, stop_time(r['day'], stop))
                        days.append({'day': r['day'], 'split': config.split_of(r['day']),
                                     **mm_metrics.day_metrics(state, r['winner'], priors, marks.get(r['day']) if stop == MAIN_STOP else None)})
                    results[stop, seed, b, fee] = days
    return results


def split_days(days, split):
    return [d for d in days if d['split'] == split]


def select(results, stop):
    """Best (b, fee) per seed by mean daily PnL on train days only."""
    chosen = {}
    for seed in forecasts.SEEDS:
        options = [(np.mean([d['pnl'] for d in split_days(days, 'train')]), b, fee)
                   for (s, name, b, fee), days in results.items() if s == stop and name == seed]
        _, chosen[seed, 'b'], chosen[seed, 'fee'] = max(options)
    return {seed: (chosen[seed, 'b'], chosen[seed, 'fee']) for seed in forecasts.SEEDS}


def run_synthetic(records, settings, marks):
    """Test days with a later Kalshi quote; informed beliefs center on that quote."""
    with (DATA / 'trader_calibration.json').open(encoding='utf-8') as f:
        fitted = json.load(f)
    fields = ('arrivals', 'threshold', 'belief_concentration', 'median_size', 'size_sigma')
    group = [r for r in records if config.split_of(r['day']) == 'test' and r['day'] in marks]
    rows = []
    b, fee = settings['noaa_calibrated'][0], SYNTHETIC_FEE
    for share in INFORMED_SHARES:
        for seed in forecasts.SEEDS:
            cfg = traders.TraderConfig(informed_share=share, fee=fee, **{k: fitted[k] for k in fields})
            daily = defaultdict(list)
            for index, r in enumerate(group):
                belief = forecasts.floor_probabilities(marks[r['day']])
                for rep in range(SYNTHETIC_REPLICATES):
                    rng = np.random.default_rng([config.BOOTSTRAP_SEED, index, rep, int(share * 100)])
                    market = LMSR(r['seeds'][seed], b)
                    state = traders.simulate_day(market, belief, cfg, rng)
                    daily[r['day']].append(mm_metrics.day_metrics(state, r['winner'], r['seeds'][seed]))
            interval = mm_metrics.bootstrap_mean([np.mean([d['pnl'] for d in v]) for v in daily.values()])
            rows.append({'split': 'test', 'informed_share': share, 'seed': seed, 'b': b, 'fee': fee, 'days': len(group),
                         'mean_pnl': interval['mean'], 'ci_low': interval['ci_low'], 'ci_high': interval['ci_high'],
                         'filled_shares': float(np.mean([d['filled_shares'] for v in daily.values() for d in v]))})
    return rows


def record_test_days(records, flow, settings):
    """Prices, inventory and outcomes per contract, and every fill, at the selected settings."""
    contracts, fills = [], []
    for r in records:
        if config.split_of(r['day']) != 'test':
            continue
        for seed in forecasts.SEEDS:
            b, fee = settings[seed]
            log = []
            state = replay_day(flow.get(r['day'], []), r['seeds'][seed], b, fee, stop_time(r['day'], MAIN_STOP), log)
            for i, ticker in enumerate(r['tickers']):
                contracts.append({'day': r['day'], 'seed': seed, 'market_ticker': ticker, 'opening_price': r['seeds'][seed][i],
                                  'closing_price': state['prices'][i], 'shares_owed': state['exposure'][i],
                                  'outcome': int(i == r['winner'])})
            for row in log:
                fills.append({'day': r['day'], 'seed': seed, 'market_ticker': r['tickers'][row.pop('outcome')], **row})
    return contracts, fills


def liquidity_rows(results, settings):
    """Effect of b on PnL, risk and responsiveness for one seed, at its selected fee."""
    seed = 'noaa_calibrated'
    fee = settings[seed][1]
    rows = []
    for b in B_GRID:
        days = results[MAIN_STOP, seed, b, fee]
        train, test = (mm_metrics.summarize(split_days(days, s)) for s in ('train', 'test'))
        rows.append({'seed': seed, 'b': b, 'fee': fee, 'train_mean_pnl': train['mean_pnl'], 'test_mean_pnl': test['mean_pnl'],
                     'test_worst_day': test['worst_day'], 'test_filled_shares': test['filled_shares'],
                     'test_peak_worst_case_loss': test['peak_worst_case_loss'], 'test_final_inventory': test['final_inventory'],
                     'test_closing_brier': test['closing_brier'], 'test_tracking_error': test['tracking_error']})
    return rows


def grid_rows(results):
    rows = []
    for (stop, seed, b, fee), days in results.items():
        for split in ('train', 'test'):
            s = mm_metrics.summarize(split_days(days, split))
            rows.append({'stop': stop, 'seed': seed, 'b': b, 'fee': fee, 'split': split, **s})
    return rows


def money(v):
    return f'-${abs(v):,.0f}' if v < 0 else f'${v:,.0f}'


def cell(i):
    return f"{money(i['mean'])} [{money(i['ci_low'])}, {money(i['ci_high'])}]"


def compare(results, settings, stop, split):
    """Per-seed day intervals and paired differences at train-selected settings."""
    series = {s: split_days(results[(stop, s) + settings[s]], split) for s in forecasts.SEEDS}
    table = {s: {'settings': settings[s], 'pnl': mm_metrics.bootstrap_mean([d['pnl'] for d in v]),
                 'summary': mm_metrics.summarize(v)} for s, v in series.items()}
    pairs = {}
    for a, b in PAIRS:
        gaps = [x['pnl'] - y['pnl'] for x, y in zip(series[a], series[b])]
        pairs[f'{a}-{b}'] = mm_metrics.bootstrap_mean(gaps)
    return table, pairs


def report(settings, tables, liquidity, records, synthetic):
    table, pairs = tables['test', MAIN_STOP]
    lines = ['# LMSR backtest (generated by market/backtest.py; do not edit by hand)', '',
             'Real Kalshi taker trades replayed as limit orders against an LMSR opened at the noon cutoff. '
             'Daily PnL for one six-bin event, in dollars. b and fee were chosen on train days only. '
             'A counterfactual estimate, not realized PnL.', '',
             f"{len(records)} days after forecast warm-up; intervals resample whole days.", '',
             '## Test days, quoting halted at noon on the target day', '',
             '| Seed | b | Fee | Daily PnL [95% CI] | Median day | Worst day | Profitable days | Shares filled/day | Peak worst-case loss | Opening Brier | Closing Brier |',
             '| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for seed, row in table.items():
        s = row['summary']
        lines.append(f"| {forecasts.LABEL[seed]} | {row['settings'][0]} | {row['settings'][1]:.2f} | {cell(row['pnl'])} | "
                     f"{money(s['median_pnl'])} | {money(s['worst_day'])} | {s['share_days_profitable']:.0%} | {s['filled_shares']:,.0f} | "
                     f"{money(s['peak_worst_case_loss'])} | {s['opening_brier']:.4f} | {s['closing_brier']:.4f} |")
    lines += ['', '| Difference in daily PnL | Mean [95% CI] | Share of replicates above 0 |', '| --- | --- | ---: |']
    for key, i in pairs.items():
        a, b = key.split('-')
        lines.append(f"| {forecasts.LABEL[a]} minus {forecasts.LABEL[b]} | {cell(i)} | {i['share_positive']:.3f} |")
    same_table, same_pairs = tables['test', 'same_settings']
    b, fee = same_table['noaa_calibrated']['settings']
    lines += ['', f'## Same settings for every seed (b {b}, fee {fee:.2f}, the Calibrated NOAA selection)', '',
              'Above, each seed has its own settings. Here only the opening forecast differs.', '',
              '| Seed | Daily PnL [95% CI] | Worst day |', '| --- | --- | ---: |']
    for seed, row in same_table.items():
        lines.append(f"| {forecasts.LABEL[seed]} | {cell(row['pnl'])} | {money(row['summary']['worst_day'])} |")
    lines += ['', '| Difference in daily PnL | Mean [95% CI] | Share of replicates above 0 |', '| --- | --- | ---: |']
    for key, i in same_pairs.items():
        first, second = key.split('-')
        lines.append(f"| {forecasts.LABEL[first]} minus {forecasts.LABEL[second]} | {cell(i)} | {i['share_positive']:.3f} |")
    close_table, _ = tables['test', 'market_close']
    lines += ['', '## Sensitivity: test days, quoting to market close', '',
              '| Seed | Daily PnL [95% CI] | Worst day |', '| --- | --- | ---: |']
    for seed, row in close_table.items():
        lines.append(f"| {forecasts.LABEL[seed]} | {cell(row['pnl'])} | {money(row['summary']['worst_day'])} |")
    lines += ['', '## Effect of b (Calibrated NOAA seed, selected fee)', '',
              'Closing Brier and tracking error are price responsiveness: tracking error is the mean distance from Kalshi at the halt.', '',
              '| b | Train PnL/day | Test PnL/day | Test worst day | Shares filled/day | Peak worst-case loss | Closing Brier | Tracking error |',
              '| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in liquidity:
        lines.append(f"| {r['b']} | {money(r['train_mean_pnl'])} | {money(r['test_mean_pnl'])} | {money(r['test_worst_day'])} | "
                     f"{r['test_filled_shares']:,.0f} | {money(r['test_peak_worst_case_loss'])} | {r['test_closing_brier']:.4f} | {r['test_tracking_error']:.4f} |")
    lines += ['', '## Synthetic traders: test-day PnL by informed share', '',
              'Informed traders hold noisy beliefs around the later Kalshi midpoint; the rest trade for reasons unrelated to the outcome. '
              f"Every seed uses b {settings['noaa_calibrated'][0]} and fee {SYNTHETIC_FEE:.2f}, so only the opening forecast differs. This is a what-if, not an estimate of real flow.", '',
              '| Informed share | ' + ' | '.join(forecasts.LABEL[s] for s in forecasts.SEEDS) + ' |',
              '| ---: |' + ' ---: |' * len(forecasts.SEEDS)]
    for share in INFORMED_SHARES:
        row = {r['seed']: r for r in synthetic if r['split'] == 'test' and r['informed_share'] == share}
        lines.append(f'| {share:.0%} | ' + ' | '.join(money(row[s]['mean_pnl']) for s in forecasts.SEEDS) + ' |')
    return '\n'.join(lines) + '\n'


def main():
    records = forecasts.day_records()
    print(f'{len(records)} days; loading flow', flush=True)
    flow = load_flow(records)
    marks = load_kalshi_marks(records)
    print(f'{sum(map(len, flow.values())):,} trades after cutoffs; replaying grid', flush=True)
    results = run_replay(records, flow, marks)
    settings = select(results, MAIN_STOP)
    tables = {(split, stop): compare(results, settings, stop, split) for split in ('train', 'test') for stop in STOPS}
    common = {s: settings['noaa_calibrated'] for s in forecasts.SEEDS}
    tables['test', 'same_settings'] = compare(results, common, MAIN_STOP, 'test')
    print('selected', settings, 'running synthetic', flush=True)
    synthetic = run_synthetic(records, settings, marks)
    liquidity = liquidity_rows(results, settings)
    contracts, fills = record_test_days(records, flow, settings)

    OUT.mkdir(parents=True, exist_ok=True)
    write_csv('results/backtest/liquidity.csv', liquidity)
    write_csv('results/backtest/contract_records.csv', contracts)
    with gzip.open(OUT / 'fills.csv.gz', 'wt', newline='', encoding='utf-8', compresslevel=9) as f:
        writer = csv.DictWriter(f, fieldnames=list(fills[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(fills)
    write_rows = lambda name, rows: write_csv(f'results/backtest/{name}', rows)
    write_rows('replay_grid.csv', grid_rows(results))
    daily = [{'seed': s, 'b': settings[s][0], 'fee': settings[s][1], 'stop': stop, **d}
             for s in forecasts.SEEDS for stop in STOPS for d in results[(stop, s) + settings[s]]]
    write_rows('replay_daily.csv', daily)
    write_rows('synthetic.csv', synthetic)
    summary = {'selected': {s: {'b': v[0], 'fee': v[1]} for s, v in settings.items()},
               'comparisons': {f'{split}/{stop}': {'seeds': t, 'pairs': p} for (split, stop), (t, p) in tables.items()},
               'days': len(records), 'test_days': len([r for r in records if config.split_of(r['day']) == 'test'])}
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    text = report(settings, tables, liquidity, records, synthetic)
    (OUT / 'backtest_tables.md').write_text(text, encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
