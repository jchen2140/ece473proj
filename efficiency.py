"""How much public weather information Kalshi prices in, how fast, and how that varies with liquidity.

Run after collect_cities.py. All times are hours from the start of the target day's climate day
(local standard midnight); the noon cutoff on D-1 is about hour -12. Writes data/results/efficiency/.

Measures, per city and pooled:
- Accuracy by hour: RMSE of the public forecast (latest NBM, bias-corrected with verified days
  through D-2), of the public forecast plus observations so far (cross-validated nowcast), and of
  the Kalshi-implied median. gamma is the slope of (high - public) on (Kalshi - public): 1 means
  every departure of the market from public data is information, 0 means it is noise.
- Event studies: NBM and GFS MOS releases (revision of the day's max forecast) and hourly station
  observations on the target day (change in the miss against the NBM hourly forecast). The path
  of the Kalshi median from 6 h before to 8 h after is regressed on the surprise. "Warranted" is
  the slope of (high - Kalshi before the event) on the surprise: how far the price should move.
  Nothing assumes a move is caused by the release; the pre-event path shows anticipation.
- Liquidity: daily volume and noon spread against each efficiency measure across cities.
Intervals resample whole days (city-days when pooled).
"""
import datetime as dt
import gzip
import json
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

OUT = DATA / 'results' / 'efficiency'
HOURS = (-12, -6, 0, 6, 9, 12, 15)
EVENT_HOURS = range(-6, 9)
LATENCY_HOURS = {'nbm': 1.25, 'gfs_mos': 3.75}  # Typical bulletin availability after the model cycle.
BIAS_WINDOW = 21
MIN_HISTORY = 10
TAIL_WIDTH = 3.0  # Assumed width (F) of the open-ended bins when reading a median off the prices.
REPLICATES = 500
RNG = np.random.default_rng(config.BOOTSTRAP_SEED)


# ---------------------------------------------------------------- loading

def read_gz(path):
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        return pd.read_csv(f)


def implied_median(probs, lows, highs):
    cum = np.cumsum(probs)
    i = int(np.searchsorted(cum, 0.5))
    i = min(i, len(probs) - 1)
    before = cum[i] - probs[i]
    frac = (0.5 - before) / probs[i] if probs[i] > 0 else 0.5
    return lows[i] + frac * (highs[i] - lows[i])


def load_city(city):
    folder = DATA / 'cities' / city.code
    markets = pd.read_csv(folder / 'markets.csv')
    candles = read_gz(folder / 'candles.csv.gz')
    days = {}
    for day, group in markets.groupby('target_date'):
        rows = group.to_dict('records')
        edges = [forecasts.bin_edges({k: ('' if pd.isna(v) else v) for k, v in r.items()}) for r in rows]
        order = np.argsort([lo for lo, _ in edges])
        rows, edges = [rows[i] for i in order], [edges[i] for i in order]
        lows = np.array([hi - 1 - TAIL_WIDTH if math.isinf(lo) else lo for lo, hi in edges])
        highs = np.array([lo + TAIL_WIDTH if math.isinf(hi) else hi for lo, hi in edges])
        winner = next(i for i, r in enumerate(rows) if r['outcome'] == 1)
        high = rows[0]['settlement_high_f']
        days[day] = {'tickers': [r['market_ticker'] for r in rows], 'lows': lows, 'highs': highs, 'winner': winner,
                     'high': float(high) if pd.notna(high) else float(np.clip(round((lows[winner] + highs[winner]) / 2), lows[winner], highs[winner])),
                     'start': cities.day_start_utc(city, dt.date.fromisoformat(day))}
    panel = []
    for day, group in candles.groupby('target_date'):
        info = days.get(day)
        if info is None:
            continue
        group = group.assign(ts=group['end_period_ts'].astype(int))
        hours = np.arange(group['ts'].min(), group['ts'].max() + 1, 3600)
        mids, spreads, volume = [], [], np.zeros(len(hours))
        for ticker in info['tickers']:
            g = group[group['market_ticker'] == ticker].drop_duplicates('ts').set_index('ts').reindex(hours)
            volume += pd.to_numeric(g['volume'], errors='coerce').fillna(0).to_numpy()
            bid, ask = g['yes_bid'].ffill().to_numpy(dtype=float), g['yes_ask'].ffill().to_numpy(dtype=float)
            ok = (bid <= ask) & ~np.isnan(bid) & ~np.isnan(ask)
            mids.append(np.where(ok, (bid + ask) / 2, np.nan))
            spreads.append(np.where(ok, ask - bid, np.nan))
        mids, spreads = np.array(mids).T, np.array(spreads).T
        complete = ~np.isnan(mids).any(axis=1) & (np.nansum(mids, axis=1) > 0)
        for j in np.flatnonzero(complete):
            probs = mids[j] / mids[j].sum()
            panel.append({'day': day, 'ts': int(hours[j]),
                          'hour': (hours[j] - info['start'].timestamp()) / 3600,
                          'median': implied_median(probs, info['lows'], info['highs']),
                          'p_win': probs[info['winner']], 'brier': float(np.mean((probs - np.eye(6)[info['winner']]) ** 2)),
                          'spread': float(np.mean(spreads[j])), 'volume': volume[j]})
    panel = pd.DataFrame(panel)
    return days, panel


def load_runs(city, name, days):
    """Per target day: (release time, max forecast, spread) for every archived run, oldest first."""
    raw = read_gz(DATA / 'cities' / city.code / f'{"nbm" if name == "nbm" else "gfs_mos"}.csv.gz')
    raw['runtime'] = pd.to_datetime(raw['runtime'], utc=True)
    raw['ftime'] = pd.to_datetime(raw['ftime'], utc=True)
    maxima = raw[raw['max_f'].notna()].copy()
    maxima['day'] = (maxima['ftime'] - pd.Timedelta(days=1)).dt.date.astype(str)
    maxima = maxima[maxima['day'].isin(days)]
    maxima['release'] = maxima['runtime'] + pd.Timedelta(hours=LATENCY_HOURS[name])
    temps = raw[raw['tmp_f'].notna()][['runtime', 'ftime', 'tmp_f']]
    return maxima.sort_values('runtime'), temps


def load_obs(city):
    obs = read_gz(DATA / 'cities' / city.code / 'asos.csv.gz')
    obs['time'] = pd.to_datetime(obs['valid_utc'], utc=True)
    return obs.sort_values('time')


# ---------------------------------------------------------------- per-hour accuracy and information shares

def asof(panel_day, ts, max_age=3 * 3600):
    """Latest complete quote at or before ts."""
    rows = panel_day[panel_day['ts'] <= ts]
    if not len(rows) or ts - rows['ts'].iloc[-1] > max_age:
        return None
    return rows.iloc[-1]


def public_forecast(runs, day, when):
    sub = runs[(runs['day'] == day) & (runs['release'] <= when)]
    return None if sub.empty else sub.iloc[-1]


def hourly_frame(city, days, panel, nbm, gfs, obs):
    """One row per (day, hour) with the Kalshi median, the bias-corrected public forecast and obs so far.

    Public = mean of the latest NBM and GFS MOS max forecasts released by that hour. In NYC, NBM alone
    trails a four-source average by about 0.008 Brier, so a single source would overstate the market's edge.
    """
    rows = []
    by_day = {d: g for d, g in panel.groupby('day')}
    obs_times = obs['time'].astype('int64').to_numpy() // 10 ** 9
    for day, info in days.items():
        if day not in by_day:
            continue
        for h in HOURS:
            when = info['start'] + dt.timedelta(hours=h)
            quote = asof(by_day[day], when.timestamp())
            run = public_forecast(nbm, day, pd.Timestamp(when))
            mos = public_forecast(gfs, day, pd.Timestamp(when))
            if quote is None or run is None:
                continue
            start = info['start'].timestamp()
            seen = obs[(obs_times >= start) & (obs_times <= when.timestamp())]['tmpf']
            point = run['max_f'] if mos is None else (run['max_f'] + mos['max_f']) / 2
            rows.append({'city': city.code, 'day': day, 'hour': h, 'high': info['high'], 'kalshi': quote['median'],
                         'kalshi_brier': quote['brier'], 'nbm': point, 'nbm_sd': run['max_sd_f'],
                         'max_so_far': float(np.round(seen.max())) if len(seen) else np.nan})
    frame = pd.DataFrame(rows)
    # Bias from the cutoff-hour forecast error over verified days through D-2.
    cutoff = frame[frame['hour'] == HOURS[0]].set_index('day')
    errors = (cutoff['high'] - cutoff['nbm']).sort_index()
    error_days = pd.to_datetime(errors.index)
    bias = {}
    for day in frame['day'].unique():
        past = errors[(error_days <= pd.Timestamp(day) - pd.Timedelta(days=2)) &
                      (error_days > pd.Timestamp(day) - pd.Timedelta(days=2 + BIAS_WINDOW))]
        bias[day] = (past.mean(), past.std()) if len(past) >= MIN_HISTORY else (np.nan, np.nan)
    frame['bias'] = frame['day'].map(lambda d: bias[d][0])
    frame['error_sd'] = frame['day'].map(lambda d: bias[d][1])
    frame['public'] = frame['nbm'] + frame['bias']
    return frame.dropna(subset=['public'])


def cv_nowcast(frame):
    """Public forecast plus observations so far, linear, 5-fold by day within each hour."""
    out = pd.Series(np.nan, index=frame.index)
    for h, g in frame.groupby('hour'):
        # Before mid-morning the running maximum is still the overnight temperature, which carries no signal.
        if h < 9 or g['max_so_far'].isna().all():
            out[g.index] = g['public']
            continue
        seen = g['max_so_far'].fillna(g['public'] - 20)
        x = np.column_stack([np.ones(len(g)), g['public'], np.maximum(g['public'], seen), seen])
        y = g['high'].to_numpy()
        days = g['day'].to_numpy()
        folds = np.array_split(np.unique(days), 5)
        for fold in folds:
            test = np.isin(days, fold)
            beta = np.linalg.lstsq(x[~test], y[~test], rcond=None)[0]
            out[g.index[test]] = x[test] @ beta
    return out


def slope(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    x = x - x.mean()
    return float((x * (y - y.mean())).sum() / (x * x).sum()) if (x * x).sum() else np.nan


def day_bootstrap(frame, statistic, cluster=('city', 'day')):
    keys = frame[list(cluster)].drop_duplicates().to_numpy()
    index = {tuple(k): g.index.to_numpy() for k, g in frame.groupby(list(cluster))}
    draws = []
    for _ in range(REPLICATES):
        pick = keys[RNG.integers(0, len(keys), len(keys))]
        draws.append(statistic(frame.loc[np.concatenate([index[tuple(k)] for k in pick])]))
    return statistic(frame), float(np.nanpercentile(draws, 2.5)), float(np.nanpercentile(draws, 97.5))


def accuracy_rows(frame, label):
    rows = []
    for h, g in frame.groupby('hour'):
        rmse = lambda col: lambda d: math.sqrt(((d['high'] - d[col]) ** 2).mean())
        gamma = lambda d: slope(d['kalshi'] - d['nowcast'], d['high'] - d['nowcast'])
        g_mean, g_lo, g_hi = day_bootstrap(g, gamma)
        rows.append({'group': label, 'hour': h, 'days': g['day'].nunique() if label != 'pooled' else len(g),
                     'rmse_public': rmse('public')(g), 'rmse_public_obs': rmse('nowcast')(g), 'rmse_kalshi': rmse('kalshi')(g),
                     'gamma': g_mean, 'gamma_ci_low': g_lo, 'gamma_ci_high': g_hi,
                     'share_of_public_error_explained': 1 - ((g['high'] - g['kalshi']) ** 2).mean() / ((g['high'] - g['nowcast']) ** 2).mean()})
    return rows


# ---------------------------------------------------------------- event studies

def event_paths(days, panel, events):
    """Kalshi median path around each event, relative to the last quote before it."""
    by_day = {d: g.set_index('ts')['median'] for d, g in panel.groupby('day')}
    rows = []
    for e in events:
        series = by_day.get(e['day'])
        if series is None:
            continue
        t0 = int(e['time'].timestamp() // 3600 * 3600)
        if t0 not in series.index:
            continue
        base = series[t0]
        path = {k: series.get(t0 + 3600 * k, np.nan) - base for k in EVENT_HOURS}
        if any(np.isnan(path[k]) for k in (1, 2, 4)):
            continue
        rows.append({**{k: v for k, v in e.items() if k != 'time'}, 'base': base, 'target_gap': days[e['day']]['high'] - base,
                     **{f'k{k}': v for k, v in path.items()}})
    return pd.DataFrame(rows)


def forecast_events(city, days, runs, panel):
    """Revision of the day's max forecast between consecutive runs, while the market is open."""
    events = []
    for day, g in runs.groupby('day'):
        g = g.sort_values('runtime')
        for prev, cur in zip(g.iloc[:-1].itertuples(), g.iloc[1:].itertuples()):
            if (cur.runtime - prev.runtime) <= pd.Timedelta(hours=6):
                events.append({'city': city.code, 'day': day, 'time': cur.release.to_pydatetime(),
                               'hour': (cur.release.to_pydatetime() - days[day]['start']).total_seconds() / 3600,
                               'surprise': cur.max_f - prev.max_f})
    return event_paths(days, panel, events)


def observation_events(city, days, nbm_temps, obs, panel):
    """Change in the miss of the hourly observation against the NBM hourly forecast, 7am-5pm on the target day."""
    events = []
    routine = obs[obs['time'].dt.minute >= 45]  # Routine METARs at :51-:56, one per hour.
    temps = nbm_temps.assign(release=nbm_temps['runtime'] + pd.Timedelta(hours=LATENCY_HOURS['nbm']))
    for day, info in days.items():
        start = pd.Timestamp(info['start'])
        window = routine[(routine['time'] >= start + pd.Timedelta(hours=6)) & (routine['time'] <= start + pd.Timedelta(hours=17))]
        latest = temps[temps['release'] <= start + pd.Timedelta(hours=6)]
        if window.empty or latest.empty:
            continue
        run = latest[latest['runtime'] == latest['runtime'].max()].set_index('ftime')['tmp_f'].sort_index()
        grid = run.index.astype('int64') // 10 ** 9
        expected = np.interp(window['time'].astype('int64') // 10 ** 9, grid, run.to_numpy())
        miss = window['tmpf'].to_numpy() - expected
        for i in range(1, len(window)):
            when = window['time'].iloc[i].to_pydatetime()
            events.append({'city': city.code, 'day': day, 'time': when, 'hour': (when - info['start']).total_seconds() / 3600,
                           'surprise': miss[i] - miss[i - 1]})
    return event_paths(days, panel, events)


def event_rows(events, label):
    """Cumulative response slopes by hour, plus the warranted response and what is left after 8 h."""
    rows = []
    for k in EVENT_HOURS:
        col = f'k{k}'
        data = events.dropna(subset=[col])
        b, lo, hi = day_bootstrap(data, lambda d: slope(d['surprise'], d[col]))
        rows.append({'group': label, 'k': k, 'events': len(data), 'slope': b, 'ci_low': lo, 'ci_high': hi})
    w, wlo, whi = day_bootstrap(events, lambda d: slope(d['surprise'], d['target_gap']))
    left = events.dropna(subset=['k8'])
    r, rlo, rhi = day_bootstrap(left, lambda d: slope(d['surprise'], d['target_gap'] - d['k8']))
    return rows, {'group': label, 'events': len(events), 'warranted': w, 'warranted_ci': [wlo, whi],
                  'left_after_8h': r, 'left_ci': [rlo, rhi]}


# ---------------------------------------------------------------- cutoff probabilities and the NYC model check

def public_brier(row, info):
    """Brier of Gaussian bins around the bias-corrected NBM forecast; tails absorb the open-ended mass."""
    sd = max(1.0, row['error_sd'] if pd.notna(row['error_sd']) else 2.5)
    p = np.array([forecasts.normal_cdf(hi, row['public'], sd) - forecasts.normal_cdf(lo, row['public'], sd)
                  for lo, hi in zip(info['lows'], info['highs'])])
    p[0] += forecasts.normal_cdf(info['lows'][0], row['public'], sd)
    p[-1] += 1 - forecasts.normal_cdf(info['highs'][-1], row['public'], sd)
    p = np.array(forecasts.floor_probabilities(list(p)))
    return float(np.mean((p - np.eye(6)[info['winner']]) ** 2)), p


def nyc_model_check(nyc_days, frame):
    """NYC: is the simple NBM forecast close to the random forest? Same days, same contracts, noon cutoff."""
    path = DATA / 'ml_forecasts.csv'
    if not path.exists():
        return None
    ml = pd.read_csv(path)
    matched = pd.read_csv(DATA / 'matched.csv')[['market_ticker', 'outcome', 'target_date', 'market_midpoint']]
    ml = ml.merge(matched, on=['market_ticker', 'target_date'])
    cut = frame[(frame['city'] == 'NYC') & (frame['hour'] == HOURS[0])].set_index('day')
    rows = []
    for day, g in ml.groupby('target_date'):
        if day not in cut.index or day not in nyc_days or g['market_midpoint'].isna().any():
            continue
        y = g['outcome'].to_numpy()
        public, _ = public_brier(cut.loc[day], nyc_days[day])
        mid = g['market_midpoint'].to_numpy()
        rows.append({'day': day, 'ml': float(np.mean((g['ml_probability'] - y) ** 2)),
                     'consensus': float(np.mean((g['consensus_probability'] - y) ** 2)), 'nbm': public,
                     'kalshi': float(np.mean((forecasts.floor_probabilities(list(mid)) - y) ** 2))})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- figures

def plot_accuracy(rows, path):
    d = pd.DataFrame([r for r in rows if r['group'] == 'pooled'])
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for col, label, color in (('rmse_public', 'Public forecast (NBM + GFS MOS, bias-corrected)', '#3b6ea8'),
                              ('rmse_public_obs', 'Public forecast + observations so far', '#2a9d8f'),
                              ('rmse_kalshi', 'Kalshi-implied median', '#444444')):
        ax.plot(d['hour'], d[col], marker='o', label=label, color=color)
    ax.axvline(0, color='#999', lw=0.8)
    ax.set_xlabel('Hours from start of target day (local standard midnight; -12 = noon cutoff)')
    ax.set_ylabel('RMSE of the daily high (Â°F)')
    ax.set_title('Where Kalshi\'s accuracy comes from (all cities pooled)')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_events(event_tables, summaries, path):
    fig, axes = plt.subplots(1, len(event_tables), figsize=(4.2 * len(event_tables), 3.8), sharey=True)
    for ax, (name, rows) in zip(np.atleast_1d(axes), event_tables.items()):
        d = pd.DataFrame([r for r in rows if r['group'] == 'pooled'])
        w = summaries[name]['warranted']
        scale = w if w and abs(w) > 1e-6 else 1.0
        ax.fill_between(d['k'], d['ci_low'] / scale, d['ci_high'] / scale, color='#3b6ea8', alpha=0.2)
        ax.plot(d['k'], d['slope'] / scale, marker='o', color='#3b6ea8')
        ax.axhline(1, color='#c05a2b', ls='--', lw=1, label='warranted move')
        ax.axhline(0, color='#555', lw=0.8)
        ax.axvline(0, color='#999', lw=0.8)
        ax.set_title(f"{name} ({summaries[name]['events']:,} events)")
        ax.set_xlabel('Hours from event')
    np.atleast_1d(axes)[0].set_ylabel('Price move / warranted move')
    np.atleast_1d(axes)[0].legend(fontsize=8)
    fig.suptitle('Kalshi response to new public information (all cities pooled)')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_liquidity(summary, path):
    d = pd.DataFrame(summary)
    panels = [('kalshi_advantage_brier', 'Brier: public âˆ’ Kalshi at noon cutoff\n(market lead over public forecast)'),
              ('nbm_absorbed_2h', 'Share of warranted NBM move\nin first 2 hours'),
              ('gamma_cutoff', 'gamma at cutoff\n(market departures that are information)')]
    fig, axes = plt.subplots(1, len(panels), figsize=(13, 3.8))
    for ax, (col, label) in zip(axes, panels):
        ax.scatter(np.log10(d['daily_volume']), d[col], color='#3b6ea8')
        for _, r in d.iterrows():
            ax.annotate(r['city'], (np.log10(r['daily_volume']), r[col]), fontsize=7, xytext=(3, 2), textcoords='offset points')
        ax.set_xlabel('log10 contracts traded per day')
        ax.set_ylabel(label, fontsize=8)
    fig.suptitle('Efficiency against liquidity across Kalshi cities')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- main

def spearman(a, b):
    return float(pd.Series(a).rank().corr(pd.Series(b).rank()))


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    (OUT / 'figures').mkdir(parents=True, exist_ok=True)
    available = [c for c in cities.CITIES if (DATA / 'cities' / c.code / 'validation.json').exists()]
    frames, events = [], {'NBM release': [], 'GFS MOS release': [], 'Station observation': []}
    summary, all_days = [], {}
    for city in available:
        days, panel = load_city(city)
        all_days[city.code] = days
        nbm, nbm_temps = load_runs(city, 'nbm', days)
        gfs, _ = load_runs(city, 'gfs_mos', days)
        obs = load_obs(city)
        frame = hourly_frame(city, days, panel, nbm, gfs, obs)
        frame['nowcast'] = cv_nowcast(frame)
        frames.append(frame)
        ev = {'NBM release': forecast_events(city, days, nbm, panel), 'GFS MOS release': forecast_events(city, days, gfs, panel),
              'Station observation': observation_events(city, days, nbm_temps, obs, panel)}
        for k, v in ev.items():
            events[k].append(v)
        cut = frame[frame['hour'] == HOURS[0]]
        public = [public_brier(r, days[r['day']])[0] for _, r in cut.iterrows()]
        daily = panel.groupby('day')['volume'].sum()
        noon = panel[(panel['hour'] >= -13) & (panel['hour'] <= -11)].groupby('day')['spread'].mean()
        nbm_rows, nbm_sum = event_rows(ev['NBM release'], city.code)
        absorbed = next(r['slope'] for r in nbm_rows if r['k'] == 2) / nbm_sum['warranted'] if nbm_sum['warranted'] else np.nan
        summary.append({'city': city.code, 'name': city.name, 'days': len(days), 'daily_volume': float(daily.mean()),
                        'noon_spread': float(noon.mean()), 'kalshi_brier_cutoff': float(cut['kalshi_brier'].mean()),
                        'public_brier_cutoff': float(np.mean(public)),
                        'kalshi_advantage_brier': float(np.mean(public) - cut['kalshi_brier'].mean()),
                        'rmse_public_cutoff': math.sqrt(((cut['high'] - cut['public']) ** 2).mean()),
                        'rmse_kalshi_cutoff': math.sqrt(((cut['high'] - cut['kalshi']) ** 2).mean()),
                        'gamma_cutoff': slope(cut['kalshi'] - cut['public'], cut['high'] - cut['public']),
                        'nbm_events': nbm_sum['events'], 'nbm_warranted': nbm_sum['warranted'], 'nbm_absorbed_2h': absorbed,
                        'nbm_left_after_8h': nbm_sum['left_after_8h']})
        print(f"{city.code}: {len(days)} days, {daily.mean():,.0f} contracts/day, Brier Kalshi {summary[-1]['kalshi_brier_cutoff']:.4f} "
              f"vs public {summary[-1]['public_brier_cutoff']:.4f}", flush=True)

    frame = pd.concat(frames, ignore_index=True)
    events = {k: pd.concat(v, ignore_index=True) for k, v in events.items()}
    accuracy = accuracy_rows(frame, 'pooled')
    for code, g in frame.groupby('city'):
        accuracy += accuracy_rows(g, code)
    event_tables, event_summaries = {}, {}
    for name, ev in events.items():
        rows, s = event_rows(ev, 'pooled')
        event_tables[name], event_summaries[name] = rows, s
        write_csv(f'results/efficiency/events_{name.split()[0].lower()}.csv', rows)
    write_csv('results/efficiency/accuracy_by_hour.csv', accuracy)
    write_csv('results/efficiency/city_summary.csv', summary)
    (OUT / 'event_summary.json').write_text(json.dumps(event_summaries, indent=2))

    # Liquidity within cities: city-demeaned log volume terciles, pooled.
    cut = frame[frame['hour'] == HOURS[0]].copy()
    volume = pd.concat([load_city_volume(c.code) for c in available], ignore_index=True)
    cut = cut.merge(volume, on=['city', 'day'], how='left')
    cut['log_volume'] = np.log10(cut['day_volume'].clip(lower=1))
    cut['relative_volume'] = cut['log_volume'] - cut.groupby('city')['log_volume'].transform('mean')
    cut['volume_tercile'] = pd.qcut(cut['relative_volume'], 3, labels=['low', 'mid', 'high'])
    cut['abs_err_kalshi'] = (cut['high'] - cut['kalshi']).abs()
    cut['abs_err_public'] = (cut['high'] - cut['public']).abs()
    within = []
    for level, g in cut.groupby('volume_tercile', observed=True):
        adv = lambda d: float((d['abs_err_public'] - d['abs_err_kalshi']).mean())
        m, lo, hi = day_bootstrap(g, adv)
        within.append({'tercile': str(level), 'city_days': len(g), 'kalshi_mae': g['abs_err_kalshi'].mean(),
                       'public_mae': g['abs_err_public'].mean(), 'kalshi_advantage_mae': m, 'ci_low': lo, 'ci_high': hi,
                       'gamma': slope(g['kalshi'] - g['public'], g['high'] - g['public'])})
    write_csv('results/efficiency/within_city_volume.csv', within)

    check = nyc_model_check(all_days.get('NYC', {}), frame)
    plot_accuracy(accuracy, OUT / 'figures' / 'accuracy_by_hour.png')
    plot_events(event_tables, event_summaries, OUT / 'figures' / 'event_study.png')
    plot_liquidity(summary, OUT / 'figures' / 'liquidity.png')
    report(summary, accuracy, event_tables, event_summaries, within, check)


def load_city_volume(code):
    candles = read_gz(DATA / 'cities' / code / 'candles.csv.gz')
    v = candles.groupby('target_date')['volume'].sum().rename('day_volume').reset_index().rename(columns={'target_date': 'day'})
    return v.assign(city=code)


def cell(m, lo, hi, fmt='{:+.2f}'):
    return f'{fmt.format(m)} [{fmt.format(lo)}, {fmt.format(hi)}]'


def report(summary, accuracy, event_tables, event_summaries, within, check):
    pooled = [r for r in accuracy if r['group'] == 'pooled']
    s = pd.DataFrame(summary)
    lines = ['# Information efficiency of Kalshi daily-high markets (generated by efficiency.py; do not edit by hand)', '',
             f'{len(s)} cities, {cities.WINDOW_START} to {cities.WINDOW_END} (every day settled on the NWS Climatological Report), '
             f'{int(s["days"].sum()):,} city-days. Hours are measured from local standard midnight starting the target day; '
             'the noon cutoff on D-1 is about hour -12. Intervals resample whole city-days.', '',
             '## 1. Where Kalshi\'s accuracy comes from (pooled)', '',
             'Public = mean of the latest NBM and GFS MOS max forecasts released by that hour, bias-corrected with verified days through D-2. '
             'Public + obs adds the station maximum observed so far (cross-validated). gamma: slope of (high âˆ’ public+obs) on '
             '(Kalshi âˆ’ public+obs); 1 means every departure of the price from public data is information. '
             '"Explained" is the share of the public+obs squared error that Kalshi removes.', '',
             '| Hour | City-days | RMSE public (Â°F) | RMSE public + obs | RMSE Kalshi | gamma [95% CI] | Explained |',
             '| ---: | ---: | ---: | ---: | ---: | --- | ---: |']
    lines += [f"| {r['hour']:+d} | {r['days']:,} | {r['rmse_public']:.2f} | {r['rmse_public_obs']:.2f} | {r['rmse_kalshi']:.2f} | "
              f"{cell(r['gamma'], r['gamma_ci_low'], r['gamma_ci_high'])} | {r['share_of_public_error_explained']:+.0%} |" for r in pooled]
    lines += ['', '## 2. How fast new public information reaches the price (pooled event studies)', '',
              'Slope of the Kalshi median move (from the last quote before the event) on the surprise, by hour. Warranted: slope of '
              '(settled high âˆ’ pre-event Kalshi median) on the surprise, i.e. how far the price should have moved. '
              'Negative hours show drift before the event (anticipation from other sources).', '']
    for name, rows in event_tables.items():
        sm = event_summaries[name]
        by_k = {r['k']: r for r in rows}
        lines += [f"**{name}** ({sm['events']:,} events). Warranted slope {cell(sm['warranted'], *sm['warranted_ci'], '{:.2f}')}; "
                  f"still unpriced after 8 h {cell(sm['left_after_8h'], *sm['left_ci'], '{:.2f}')}.", '',
                  '| Hours from event | -6 | -3 | -1 | +1 | +2 | +4 | +8 |', '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
                  '| Cumulative slope | ' + ' | '.join(f"{by_k[k]['slope']:+.2f}" for k in (-6, -3, -1, 1, 2, 4, 8)) + ' |',
                  '| Share of warranted | ' + ' | '.join(f"{by_k[k]['slope'] / sm['warranted']:+.0%}" if sm['warranted'] else '' for k in (-6, -3, -1, 1, 2, 4, 8)) + ' |', '']
    lines += ['## 3. Liquidity across cities', '',
              'Daily volume is contracts traded over each event\'s life. Advantage = public Brier âˆ’ Kalshi Brier at the noon cutoff '
              '(six-bin average; public uses Gaussian bins around the bias-corrected NBM + GFS MOS mean).', '',
              '| City | Days | Contracts/day | Noon spread | Kalshi Brier | Public Brier | Advantage | RMSE Kalshi | RMSE public | gamma | NBM absorbed in 2 h |',
              '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in sorted(summary, key=lambda r: -r['daily_volume']):
        lines.append(f"| {r['name']} | {r['days']} | {r['daily_volume']:,.0f} | {r['noon_spread']:.3f} | {r['kalshi_brier_cutoff']:.4f} | "
                     f"{r['public_brier_cutoff']:.4f} | {r['kalshi_advantage_brier']:+.4f} | {r['rmse_kalshi_cutoff']:.2f} | "
                     f"{r['rmse_public_cutoff']:.2f} | {r['gamma_cutoff']:.2f} | {r['nbm_absorbed_2h']:.0%} |")
    v = np.log10(s['daily_volume'])
    lines += ['', f"Spearman rank correlation with log volume across {len(s)} cities: advantage {spearman(v, s['kalshi_advantage_brier']):+.2f}, "
              f"gamma {spearman(v, s['gamma_cutoff']):+.2f}, NBM absorbed in 2 h {spearman(v, s['nbm_absorbed_2h']):+.2f}, "
              f"noon spread {spearman(v, s['noon_spread']):+.2f}.", '',
              '**Within cities** (days split by volume relative to that city\'s average; mean absolute error at the cutoff):', '',
              '| Volume tercile | City-days | Kalshi MAE | Public MAE | Kalshi advantage [95% CI] | gamma |', '| --- | ---: | ---: | ---: | --- | ---: |']
    lines += [f"| {r['tercile']} | {r['city_days']} | {r['kalshi_mae']:.2f} | {r['public_mae']:.2f} | "
              f"{cell(r['kalshi_advantage_mae'], r['ci_low'], r['ci_high'])} | {r['gamma']:.2f} |" for r in within]
    if check is not None and len(check):
        lines += ['', '## 4. Is the ML model needed for this? (NYC, same days)', '',
                  f"Mean Brier per contract at the noon cutoff over {len(check)} NYC days in the window:", '',
                  '| Forecast | Brier |', '| --- | ---: |']
        lines += [f"| {label} | {check[col].mean():.4f} |" for col, label in (('kalshi', 'Kalshi midpoint'), ('ml', 'ML random forest'),
                                                                          ('consensus', 'Four-source consensus'), ('nbm', 'Bias-corrected NBM + GFS MOS (used here)'))]
    lines += ['', '## Caveats', '',
              '- Release times are assumed (NBM text about 1 h 15 min after the cycle, GFS MOS about 3 h 45 min); hourly candles '
              'blur anything shorter than an hour.',
              '- IEM archives 4 NBM runs a day, but NBM runs hourly, and the raw GFS, HRRR and ECMWF fields it blends appear '
              'earlier. Drift before an archived release is therefore expected and is not evidence of private information.',
              '- The observation surprise is a first difference, so it is negatively autocorrelated; its pre-event path is '
              'partly mechanical. The post-event path is the measure that matters.',
              '- Volume is not exogenous: busy days are the uncertain ones (higher error for both forecasts). Cross-city '
              'correlations use 15 points; |rho| below about 0.52 is not significant at 5%.',
              '- The settlement high matches the station\'s own ASOS maximum within 1°F on 76% (Denver) to 96% (Las Vegas) '
              'of days; see data/cities/*/validation.json.']
    text = '\n'.join(lines) + '\n'
    (OUT / 'efficiency_tables.md').write_text(text, encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
