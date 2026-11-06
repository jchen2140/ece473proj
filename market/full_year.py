"""Full-year, seven-city walk-forward test of the domain market-maker ladder.

The six-city study in market/makers.py trains on Apr-May 2026 and tests Jun-Aug 2026, because the
newest series start in April. Seven older series traded under NWS rules for the whole year
(cities.FULL_YEAR_CITIES, Aug 14, 2025 - Aug 13, 2026), so this module replays a year of real Kalshi
taker flow in each and re-chooses every setting once a month from earlier days only:

- Test months: Oct 2025 - Aug 2026, starting later in a city that has fewer than 20 strict days
  (Jessie's minimum for the sensor margin) before October. Each month's settings (b, fee per maker, and the conservative
  sensor margin) are fit on every eligible day whose high was final at the first quote of that month,
  i.e. days up to two days before the month starts (expanding window, Aug 14 onward).
- Makers: the ladder of market/makers.py (generic, forecast, forecast_live), Jessie's conservative
  truncation (forecast_live_cons, margin refit monthly), the Kalshi references, and two ablations
  (halt at noon, confidence-scaled b). The throttle is left out: it lost money and overfit in the
  six-city study.
- forecast_live_filtered (exploratory): truncation on a 5-minute rolling median of the 1-minute readings.
  Every NYC day where truncation wrote off the winner was a lone 1-minute spike, which a margin handles
  by shifting every reading. The rule was fixed from the NYC diagnosis alone and recorded in
  spike_filter_protocol.json before the other six cities were replayed.
- Open-ended tail bins keep their true infinite bounds for every maker (as in market/conservative.py).
  forecast.public.load_city gives tails finite placeholder edges for reading medians off prices; the
  live tilt must not treat those as knock-out edges, or the top bin can be written off on a hot day
  when it is the winner.
- Every maker is replayed on the same days (every day with trade flow), so a difference between
  makers comes from the maker alone. A live maker acts on whatever 1-minute feed exists, as a real one
  would. Jessie's stricter rule (>= 80% one-minute coverage and an official high) is used to fit the
  sensor margin, and section 1b reports the comparison again on only those days.

market/makers.py and market/conservative.py are imported unchanged: their SHA-256 hashes are frozen
by data/results/conservative_truncation/cross_city_protocol.json. Run after
`python -m collect.city_markets --full-year` and `python -m collect.realtime --full-year`.
Writes data/results/full_year/. A counterfactual replay estimate, not realized PnL.
"""
import argparse
import json
import datetime as dt
import math
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import cities
from common import DATA, write_csv
from collect.city_markets import FULL_YEAR_OUT
from collect.realtime import FULL_YEAR_RAW
from forecast import probabilities as forecasts
from forecast import public
from market import conservative, makers
from market import metrics as mm_metrics

OUT = DATA / 'results' / 'full_year'
FIRST_TEST_MONTH = dt.date(2025, 10, 1)
B_GRID = (10, 30, 100, 300, 1000)
FEE_GRID = makers.FEE_GRID
FRONTIER_FEE = 0.05  # The risk-vs-b frontier holds the fee fixed so only liquidity moves.
MAKERS = ('generic', 'forecast', 'forecast_live', 'forecast_live_cons', 'forecast_live_filtered', 'kalshi', 'kalshi_live',
          'forecast_noon', 'forecast_confidence')
SPIKE_WINDOW = '5min'  # Fixed from the NYC diagnosis before any other city ran; see spike_filter_protocol.json.
LABEL = {'generic': 'L0 generic (uniform prior)', 'forecast': 'L1 + weather forecast prior',
         'forecast_live': 'L2 + live running-max truncation', 'forecast_live_cons': 'L2c + conservative truncation (sensor margin)',
         'forecast_live_filtered': 'L2f + truncation on spike-filtered readings',
         'kalshi': 'Ref: Kalshi noon prior', 'kalshi_live': 'Ref: Kalshi prior + live truncation',
         'forecast_noon': 'Ablation: forecast, halt at noon', 'forecast_confidence': 'Ablation: forecast, confidence-scaled b'}
STEPS = (('forecast', 'generic', 'weather forecast prior'),
         ('forecast_live', 'forecast', 'live running-max truncation'),
         ('forecast_live_cons', 'forecast', 'conservative truncation'),
         ('forecast_live_cons', 'forecast_live', 'sensor margin vs plain truncation'),
         ('forecast_live_cons', 'generic', 'both domain ideas (conservative)'),
         ('forecast_live_filtered', 'forecast', 'spike-filtered truncation'),
         ('forecast_live_filtered', 'forecast_live', 'spike filter vs plain truncation'),
         ('kalshi_live', 'kalshi', 'live truncation on the crowd prior'),
         ('forecast_noon', 'forecast', 'halting at noon'),
         ('forecast_confidence', 'forecast', 'confidence-scaled b'))
SAME_BASE = 'forecast'
SEASONS = {12: 'winter', 1: 'winter', 2: 'winter', 3: 'spring', 4: 'spring', 5: 'spring',
           6: 'summer', 7: 'summer', 8: 'summer', 9: 'fall', 10: 'fall', 11: 'fall'}


# ---------------------------------------------------------------- inputs

def true_bounds(code):
    """Per day: real contract edges (+/-inf tails) and the official high, from data/cities_full/."""
    markets = pd.read_csv(FULL_YEAR_OUT / code / 'markets.csv', dtype=str, keep_default_na=False)
    out = {}
    for day, group in markets.groupby('target_date'):
        edges = sorted(forecasts.bin_edges(r) for r in group.to_dict('records'))
        highs = {float(v) for v in group['settlement_high_f'] if v not in ('', 'nan')}
        if len(highs) > 1:
            raise ValueError(f'{code} {day}: conflicting official highs')
        out[day] = {'lows': np.array([e[0] for e in edges]), 'highs': np.array([e[1] for e in edges]),
                    'official_high_f': next(iter(highs)) if highs else None}
    return out


def load_days(code):
    """market/makers.day_inputs on the full-year data with true bounds. Returns (days with trade flow, audit
    rows); audit 'eligible' is market/conservative.prepare_days' strict rule, used to fit the margin."""
    city = cities.BY_CODE[code]
    saved = makers.RAW, makers.TRAIN_END
    loaders = public.load_city, public.load_runs, public.load_obs
    # day_inputs reads module-level paths; point them at the full-year data for this call only.
    makers.RAW, makers.TRAIN_END = FULL_YEAR_RAW, dt.date.max
    public.load_city = lambda c, root=None: loaders[0](c, FULL_YEAR_OUT)
    public.load_runs = lambda c, name, days, root=None: loaders[1](c, name, days, FULL_YEAR_OUT)
    public.load_obs = lambda c, root=None: loaders[2](c, FULL_YEAR_OUT)
    try:
        days = makers.day_inputs(code)
    finally:
        makers.RAW, makers.TRAIN_END = saved
        public.load_city, public.load_runs, public.load_obs = loaders
    bounds = true_bounds(code)
    replayed, audit = [], []
    for d in days:
        info = bounds[d['day']]
        d = {**d, 'lows': info['lows'], 'highs': info['highs'], 'city': code}
        coverage = len(np.unique(np.floor(d['minute_ts'] / 60))) / 1440
        sensor = float(d['running_max'].max()) if len(d['running_max']) else None
        official = info['official_high_f']
        reason = ('insufficient_minute_coverage' if coverage < conservative.MIN_COVERAGE else
                  'missing_official_temperature' if official is None else
                  'no_trade_flow' if not d['flow'] else '')
        audit.append({'city': code, 'day': d['day'], 'coverage_fraction': coverage, 'sensor_high_f': sensor,
                      'official_high_f': official,
                      'sensor_minus_official_f': sensor - official if sensor is not None and official is not None else None,
                      'eligible': int(not reason), 'exclusion_reason': reason, 'replayed': int(bool(d['flow']))})
        if d['flow']:
            replayed.append({**d, 'strict': int(not reason)})
    return replayed, audit


# ---------------------------------------------------------------- walk-forward

def filtered_running_max(code, days):
    """Running maximum of the 5-minute rolling median of the 1-minute readings, aligned with each day's minute_ts.

    Every NYC day where live truncation wrote off the winner was a single-minute spike (e.g. Jan 14, 2026: one
    reading of 60F against an official 52F and an hourly maximum of 52F). A rolling median ignores a lone spike
    but follows any sustained rise. The window ends at the reading time, so the filter uses no future data.
    """
    minute = public.read_gz(FULL_YEAR_RAW / code / 'asos1min.csv.gz')
    minute['ts'] = pd.to_datetime(minute['valid_utc'], utc=True)
    series = minute.sort_values('ts').drop_duplicates('ts').set_index('ts')['tmpf'].astype(float)
    median = series.rolling(SPIKE_WINDOW).median()
    by_ts = dict(zip(median.index.astype('int64') // 10 ** 9, median.to_numpy()))
    for d in days:
        values = np.array([by_ts.get(int(t), np.nan) for t in d['minute_ts']])
        d['running_max_filtered'] = np.fmax.accumulate(values) if len(values) else values


def filtered_tilt(day, ts):
    """market/makers.tilt on the spike-filtered running maximum."""
    weights = np.ones(len(day['highs']))
    i = np.searchsorted(day['minute_ts'], ts - makers.FEED_DELAY, side='right')
    if i == 0 or not np.isfinite(day['running_max_filtered'][i - 1]):
        return weights, None
    m = day['running_max_filtered'][i - 1]
    finite = np.isfinite(day['highs'])
    weights[finite & (day['highs'] < m)] = makers.TOUCHED_WEIGHT
    weights[finite & (day['highs'] < m - 1)] = makers.DEAD_WEIGHT
    return weights, m


def month_of(day):
    return day[:7]


def fit_cutoff(month):
    """Latest day whose high is final at the first quote of `month` (noon on the day before day 1)."""
    return (dt.date.fromisoformat(month + '-01') - dt.timedelta(days=2)).isoformat()


def test_months(days):
    first = FIRST_TEST_MONTH.isoformat()[:7]
    return sorted({month_of(d['day']) for d in days if month_of(d['day']) >= first})


def fit_margin(audit, cutoff):
    """Jessie's frozen rule on eligible days up to cutoff: nearest-rank 99th percentile of positive
    sensor-minus-official errors, rounded up to 0.5F."""
    rows = [{**r, 'split': 'train'} for r in audit if r['eligible'] and r['day'] <= cutoff]
    return conservative.fit_margin(rows, cutoff)


def sd_reference(days):
    """Typical forecast error sd for confidence scaling, from the days before the first test month."""
    return float(np.median([d['sd'] for d in days if d['day'] < FIRST_TEST_MONTH.isoformat()]))


def replay_one(day, maker, b, fee, sd_ref, margin):
    if maker == 'forecast_live_cons':
        result = makers.replay(day, 'forecast_live', b, fee, 1.0, sd_ref,
                               tilt_function=lambda d, ts: conservative.conservative_tilt(d, ts, margin))
    elif maker == 'forecast_live_filtered':
        result = makers.replay(day, 'forecast_live', b, fee, 1.0, sd_ref, tilt_function=filtered_tilt)
    else:
        result = makers.replay(day, maker, b, fee, 1.0, sd_ref)
    prior = day['priors'][makers.SPECS[maker if maker in makers.SPECS else 'forecast_live']['prior']]
    # Loss bound of the opening prior if the winner had won with no tilt: b * ln(1 / prior_w).
    result['opening_bound'] = b * math.log(1 / prior[day['winner']])
    result['max_opening_bound'] = b * math.log(1 / prior.min())
    return result


def city_grid(code):
    """Replay every maker, b and fee on every eligible day. The conservative maker is replayed once per
    distinct monthly margin. Returns (days, audit, margins by month, grid rows)."""
    days, audit = load_days(code)
    filtered_running_max(code, days)
    sd_ref = sd_reference(days)
    margins = {}
    for m in test_months(days):
        try:
            margins[m] = fit_margin(audit, fit_cutoff(m))
        except ValueError:
            # Jessie's rule needs 20 strict days; a city's test period starts once it has them.
            if margins:
                raise
    if not margins:
        raise ValueError(f'{code}: never reaches 20 strict days to fit the sensor margin')
    rows = []
    for maker in MAKERS:
        margin_values = sorted({v['margin_f'] for v in margins.values()}) if maker == 'forecast_live_cons' else [0.0]
        for margin in margin_values:
            for b in B_GRID:
                for fee in FEE_GRID:
                    for d in days:
                        r = replay_one(d, maker, b, fee, sd_ref, margin)
                        rows.append({'city': code, 'day': d['day'], 'strict': d['strict'], 'maker': maker, 'b': b, 'fee': fee,
                                     'margin_f': margin, **{k: v for k, v in r.items() if not isinstance(v, (list, str))}})
        print(f'  {code} {maker} done', flush=True)
    return days, audit, margins, rows


def choose(grid, maker, cutoff, margin=0.0):
    """Best (b, fee) by mean PnL over days up to cutoff; ties go to the smaller b and fee."""
    train = grid[(grid['maker'] == maker) & (grid['day'] <= cutoff) & (grid['margin_f'] == margin)]
    means = train.groupby(['b', 'fee'])['pnl'].mean().reset_index()
    means['key'] = list(zip(means['pnl'], -means['b'], -means['fee']))
    best = means.loc[means['key'].idxmax()]
    return float(best['b']), float(best['fee']), int(train['day'].nunique())


def walk_forward(code, grid, margins):
    """Per test month, settings fit on earlier days; returns the test-day rows for both comparisons."""
    out, settings = [], []
    for month, fit in margins.items():
        cutoff = fit_cutoff(month)
        margin = fit['margin_f']
        same_b, same_fee, n_train = choose(grid, SAME_BASE, cutoff)
        for maker in MAKERS:
            m = margin if maker == 'forecast_live_cons' else 0.0
            own_b, own_fee, _ = choose(grid, maker, cutoff, m)
            settings.append({'city': code, 'month': month, 'maker': maker, 'fit_through': cutoff, 'train_days': n_train,
                             'own_b': own_b, 'own_fee': own_fee, 'same_b': same_b, 'same_fee': same_fee, 'margin_f': m})
            pick = (grid['maker'] == maker) & (grid['margin_f'] == m) & (grid['day'].str[:7] == month)
            for mode, b, fee in (('own', own_b, own_fee), ('same', same_b, same_fee)):
                rows = grid[pick & (grid['b'] == b) & (grid['fee'] == fee)]
                out.extend({**r, 'mode': mode, 'month': month} for r in rows.to_dict('records'))
    return out, settings


# ---------------------------------------------------------------- report

def money(v):
    return f'-${abs(v):,.0f}' if v < 0 else f'${v:,.0f}'


def pooled_steps(test, group_cols=()):
    """Same-settings gaps per step: per-city means and a pooled, city-sd-scaled bootstrap."""
    same = test[test['mode'] == 'same'].pivot_table(index=['city', 'day'], columns='maker', values='pnl')
    rows = []
    for a, b, idea in STEPS:
        gaps = (same[a] - same[b]).rename('gap').reset_index()
        gaps['season'] = gaps['day'].map(lambda d: SEASONS[int(d[5:7])])
        groups = gaps.groupby(list(group_cols)) if group_cols else [((), gaps)]
        for key, g in groups:
            per_city = g.groupby('city')['gap'].mean()
            scale = gaps.groupby('city')['gap'].std().replace(0, 1.0)  # Full-year sd, so seasons share one scale.
            pooled = mm_metrics.bootstrap_mean((g['gap'] / g['city'].map(scale)).to_numpy())
            rows.append({'idea': idea, 'step': f'{a} - {b}', **dict(zip(group_cols, key if isinstance(key, tuple) else (key,))),
                         'city_days': len(g), 'cities_positive': int((per_city > 0).sum()), 'cities': len(per_city),
                         **{f'{c}_mean': v for c, v in per_city.items()}, 'pooled_scaled_mean': pooled['mean'],
                         'pooled_ci_low': pooled['ci_low'], 'pooled_ci_high': pooled['ci_high'],
                         'pooled_share_positive': pooled['share_positive']})
    return rows


def own_summary(test):
    rows = []
    for (code, maker), g in test[test['mode'] == 'own'].groupby(['city', 'maker'], sort=False):
        days = g.to_dict('records')
        s = mm_metrics.summarize(days)
        ci = mm_metrics.bootstrap_mean(g['pnl'].to_numpy())
        rows.append({'city': code, 'maker': maker, 'days': s['days'], 'mean_pnl': s['mean_pnl'], 'ci_low': ci['ci_low'],
                     'ci_high': ci['ci_high'], 'worst_day': s['worst_day'], 'p05_pnl': s['p05_pnl'],
                     'profitable_days': s['share_days_profitable'], 'filled_shares': s['filled_shares'],
                     'pnl_per_share': s['pnl_per_share'], 'opening_brier': s['opening_brier'],
                     'peak_worst_case_loss': s['peak_worst_case_loss'], 'median_b': float(g['b'].median())})
    return rows


def step_table(rows, codes):
    lines = ['| Idea | Step | ' + ' | '.join(codes) + ' | Cities positive | City-days | Pooled (sd units) [95% CI] |',
             '| --- | --- |' + ' ---: |' * len(codes) + ' ---: | ---: | --- |']
    for r in rows:
        lines.append(f"| {r['idea']} | {r['step']} | " + ' | '.join(money(r.get(f'{c}_mean', float('nan'))) for c in codes) +
                     f" | {r['cities_positive']}/{r['cities']} | {r['city_days']:,} | {r['pooled_scaled_mean']:+.3f} "
                     f"[{r['pooled_ci_low']:+.3f}, {r['pooled_ci_high']:+.3f}] |")
    return lines


def write_report(test, settings, audit, step_rows, strict_rows, season_rows, own_rows):
    codes = sorted(test['city'].unique(), key=cities.FULL_YEAR_CITIES.index)
    elig = pd.DataFrame(audit)
    lines = ['# Full-year walk-forward market-maker ladder (generated by market/full_year.py; do not edit by hand)', '',
             f'Real Kalshi taker trades replayed against LMSR makers in {len(codes)} cities, {cities.FULL_YEAR_START} to '
             f'{cities.WINDOW_END}. Test months {test["month"].min()} to {test["month"].max()}: each month\'s settings are fit '
             'on every eligible earlier day whose high was final at that month\'s first quote. '
             f'{test.drop_duplicates(["city", "day"]).shape[0]:,} test city-days. Daily PnL in dollars for one six-bin event; '
             'a counterfactual estimate, not realized PnL.', '',
             'Days replayed (trade flow after the cutoff) / strict days (also >= '
             f'{conservative.MIN_COVERAGE:.0%} one-minute coverage and an official high, used to fit the sensor margin), '
             'out of all days with a forecast and a quote: '
             + ', '.join(f"{c} {int(g['replayed'].sum())}/{int(g['eligible'].sum())}/{len(g)}" for c, g in elig.groupby('city', sort=False))
             + '.', '',
             '## 1. What each idea adds (same settings for every maker)', '',
             f"Each month every maker uses the {SAME_BASE} maker's walk-forward b and fee, so a difference comes from the idea "
             'alone. Mean test-day PnL gain per city; pooled = each city\'s daily gaps divided by their own standard deviation, '
             'then all city-days resampled.', '',
             *step_table(step_rows, codes), '',
             '### 1b. Same comparison on strict days only', '',
             'Only days that also pass the conservative-truncation rule (>= 80% one-minute coverage and an official high), '
             'as in market/conservative.py.', '', *step_table(strict_rows, codes)]
    lines += ['', '## 2. By season (same settings, pooled over cities)', '',
              '| Idea | ' + ' | '.join(s.capitalize() for s in ('fall', 'winter', 'spring', 'summer')) + ' |',
              '| --- |' + ' --- |' * 4]
    for idea in dict.fromkeys(r['idea'] for r in season_rows):
        cells = []
        for season in ('fall', 'winter', 'spring', 'summer'):
            r = next((x for x in season_rows if x['idea'] == idea and x['season'] == season), None)
            cells.append(f"{r['pooled_scaled_mean']:+.2f} [{r['pooled_ci_low']:+.2f}, {r['pooled_ci_high']:+.2f}] "
                         f"({r['cities_positive']}/{r['cities']})" if r else '')
        lines.append(f'| {idea} | ' + ' | '.join(cells) + ' |')
    lines += ['', '## 3. Each maker with its own walk-forward settings', '',
              '| City | Maker | Median b | Days | Test PnL/day [95% CI] | Worst day | 5th pct day | Profitable days | Shares/day | '
              'PnL/share | Open Brier | Peak worst-case loss |',
              '| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in own_rows:
        lines.append(f"| {r['city']} | {LABEL[r['maker']]} | {r['median_b']:g} | {r['days']} | {money(r['mean_pnl'])} "
                     f"[{money(r['ci_low'])}, {money(r['ci_high'])}] | {money(r['worst_day'])} | {money(r['p05_pnl'])} | "
                     f"{r['profitable_days']:.0%} | {r['filled_shares']:,.0f} | {r['pnl_per_share']:+.3f} | "
                     f"{r['opening_brier']:.4f} | {money(r['peak_worst_case_loss'])} |")
    margins = pd.DataFrame(settings).query("maker == 'forecast_live_cons'")
    lines += ['', '## 4. Fitted sensor margins (conservative truncation)', '',
              '| City | ' + ' | '.join(sorted(margins['month'].unique())) + ' |', '| --- |' + ' ---: |' * margins['month'].nunique()]
    for code in codes:
        g = margins[margins['city'] == code].set_index('month')['margin_f']
        lines.append(f'| {code} | ' + ' | '.join(f'{g.get(m, float("nan")):g}' for m in sorted(margins['month'].unique())) + ' |')
    lines += spike_section(test)
    lines += ['', '## Caveats', '',
              '- Replay: a real trade fills only if the maker would have been at least as good, at the taker\'s own price. Trades the '
              'maker would have attracted but Kalshi did not are missing, and traders might have reacted to these quotes.',
              f'- Live makers act on the 1-minute reading from {makers.FEED_DELAY // 60} minutes earlier, standing in for the public feed.',
              '- Every setting is chosen from earlier days only, but the grids, the walk-forward rule and the maker list were fixed '
              'after seeing the six-city results, so this is a stronger out-of-sample check, not a blind one.',
              '- Fees are a hypothetical maker charge, not Kalshi\'s fee schedule.']
    text = '\n'.join(lines) + '\n'
    (OUT / 'full_year_tables.md').write_text(text, encoding='utf-8')
    return text


def worst_share_mean(x, share=0.05):
    x = np.sort(np.asarray(x, dtype=float))
    return x[:int(np.ceil(share * len(x)))].mean()


def spike_section(test):
    """The pre-registered held-out check of the spike filter (spike_filter_protocol.json), plus each truncation's tail."""
    protocol = json.loads((OUT / 'spike_filter_protocol.json').read_text())
    held = [c for c in protocol['cities_held_out'] if c in set(test['city'])]
    same = test[test['mode'] == 'same'].pivot_table(index=['city', 'day'], columns='maker', values='pnl')
    rows = []
    for a, b in (('forecast_live_filtered', 'forecast_live'), ('forecast_live_filtered', 'forecast')):
        gaps = (same[a] - same[b]).rename('gap').reset_index()
        gaps = gaps[gaps['city'].isin(held)]
        scale = gaps.groupby('city')['gap'].std().replace(0, 1.0)
        pooled = mm_metrics.bootstrap_mean((gaps['gap'] / gaps['city'].map(scale)).to_numpy())
        dollars = mm_metrics.bootstrap_mean(gaps['gap'].to_numpy())
        rows.append({'comparison': f'{a} - {b}', 'cities': len(held),
                     'cities_positive': int((gaps.groupby('city')['gap'].mean() > 0).sum()), 'city_days': len(gaps),
                     'mean_gap': dollars['mean'], 'gap_ci_low': dollars['ci_low'], 'gap_ci_high': dollars['ci_high'],
                     'pooled_scaled_mean': pooled['mean'], 'pooled_ci_low': pooled['ci_low'], 'pooled_ci_high': pooled['ci_high']})
    write_csv('results/full_year/spike_filter_heldout.csv', rows)
    tails = []
    tail_makers = ('forecast', 'forecast_live', 'forecast_live_cons', 'forecast_live_filtered')
    for code in [c for c in cities.FULL_YEAR_CITIES if c in set(test['city'])]:
        for m in tail_makers:
            x = same.loc[code][m].dropna().to_numpy()
            tails.append({'city': code, 'maker': m, 'days': len(x), 'mean_pnl': x.mean(), 'worst_day': x.min(),
                          'worst_5pct_mean': worst_share_mean(x)})
    write_csv('results/full_year/same_settings_tails.csv', tails)
    lines = ['', '## 5. Spike filter: pre-registered held-out check', '',
             f"Rule frozen {protocol['created_utc'][:10]} after looking at NYC only (`spike_filter_protocol.json`). "
             f"Held-out cities: {', '.join(held)}. Same settings for every maker.", '',
             '| Comparison | Cities positive | City-days | Mean gap $/day [95% CI] | Pooled (sd units) [95% CI] |',
             '| --- | ---: | ---: | --- | --- |']
    for r in rows:
        lines.append(f"| {r['comparison']} | {r['cities_positive']}/{r['cities']} | {r['city_days']:,} | {money(r['mean_gap'])} "
                     f"[{money(r['gap_ci_low'])}, {money(r['gap_ci_high'])}] | {r['pooled_scaled_mean']:+.3f} "
                     f"[{r['pooled_ci_low']:+.3f}, {r['pooled_ci_high']:+.3f}] |")
    lines += ['', '### Tails at the same settings (mean of the worst 5% of test days / worst day)', '',
              '| City | ' + ' | '.join(LABEL[m].split(' (')[0] for m in tail_makers) + ' |', '| --- |' + ' ---: |' * len(tail_makers)]
    t = pd.DataFrame(tails)
    for code in t['city'].unique():
        g = t[t['city'] == code].set_index('maker')
        lines.append(f'| {code} | ' + ' | '.join(f"{money(g.loc[m, 'worst_5pct_mean'])} / {money(g.loc[m, 'worst_day'])}"
                                                  for m in tail_makers) + ' |')
    return lines


def plot_ladder(step_rows, season_rows, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ideas = [r['idea'] for r in step_rows]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    y = np.arange(len(ideas))[::-1]
    offsets = {'all year': 0.0, 'fall': -0.3, 'winter': -0.15, 'spring': 0.15, 'summer': 0.3}
    colors = {'all year': '#1f2933', 'fall': '#c05a2b', 'winter': '#2a78d6', 'spring': '#1baf7a', 'summer': '#eda100'}
    series = [('all year', step_rows)] + [(s, [r for r in season_rows if r['season'] == s]) for s in ('fall', 'winter', 'spring', 'summer')]
    for name, rows in series:
        by = {r['idea']: r for r in rows}
        xs = [by[i]['pooled_scaled_mean'] for i in ideas]
        lo = [by[i]['pooled_scaled_mean'] - by[i]['pooled_ci_low'] for i in ideas]
        hi = [by[i]['pooled_ci_high'] - by[i]['pooled_scaled_mean'] for i in ideas]
        ax.errorbar(xs, y + offsets[name] * (0 if name == 'all year' else 1), xerr=[lo, hi], fmt='o',
                    ms=7 if name == 'all year' else 4, color=colors[name], ecolor=colors[name], elinewidth=1.6 if name == 'all year' else 0.8,
                    capsize=0, label=name, zorder=3 if name == 'all year' else 2)
    ax.axvline(0, color='#52514e', lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(ideas, fontsize=8)
    ax.set_xlabel('Test-day PnL gain, pooled over cities (city sd units, 95% CI)')
    ax.set_title('What each idea adds, walk-forward over a full year (same settings)', loc='left', fontsize=10)
    ax.legend(frameon=False, fontsize=7, ncol=5, loc='upper left', bbox_to_anchor=(0, -0.13))
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cities', nargs='*', default=cities.FULL_YEAR_CITIES)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--report-only', action='store_true',
                        help='rebuild the tables and figure from the saved test_daily/settings/eligibility files')
    args = parser.parse_args()
    (OUT / 'figures').mkdir(parents=True, exist_ok=True)
    if args.report_only:
        test = pd.read_csv(OUT / 'test_daily.csv.gz', dtype={'day': str})
        settings = pd.read_csv(OUT / 'settings.csv').to_dict('records')
        audit = pd.read_csv(OUT / 'eligibility.csv', dtype={'day': str}).to_dict('records')
        return report(test, settings, audit)
    codes = [c for c in args.cities if (FULL_YEAR_RAW / c / 'trades.csv.gz').exists()]
    test, settings, audit, grid_all = [], [], [], []
    with ProcessPoolExecutor(args.workers) as pool:
        for code, (days, city_audit, margins, rows) in zip(codes, pool.map(city_grid, codes)):
            grid = pd.DataFrame(rows)
            grid_all.append(grid)
            t, s = walk_forward(code, grid, margins)
            test += t
            settings += s
            audit += city_audit
    grid = pd.concat(grid_all)
    test = pd.DataFrame(test)
    grid.to_csv(OUT / 'grid_daily.csv.gz', index=False, compression='gzip')
    test.to_csv(OUT / 'test_daily.csv.gz', index=False, compression='gzip')
    write_csv('results/full_year/settings.csv', settings)
    write_csv('results/full_year/eligibility.csv', audit)
    report(test, settings, audit)


def report(test, settings, audit):
    step_rows = pooled_steps(test)
    strict_rows = pooled_steps(test[test['strict'] == 1])
    write_csv('results/full_year/same_settings_strict_days.csv', strict_rows)
    season_rows = pooled_steps(test, ('season',))
    own_rows = own_summary(test)
    write_csv('results/full_year/same_settings.csv', step_rows)
    # Cities start testing in different months, so a season row may lack some cities' columns.
    pd.DataFrame(season_rows).to_csv(OUT / 'same_settings_by_season.csv', index=False)
    write_csv('results/full_year/own_settings.csv', own_rows)
    plot_ladder(step_rows, season_rows, OUT / 'figures' / 'ladder_full_year.png')
    print(write_report(test, settings, audit, step_rows, strict_rows, season_rows, own_rows))


if __name__ == '__main__':
    main()
