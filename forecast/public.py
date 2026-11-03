"""City data loaders and the public forecast: Kalshi quote panels, NWS MOS runs, ASOS observations.

The public forecast is the mean of the latest NBM and GFS MOS max forecasts released by a given hour,
bias-corrected with verified days through D-2, turned into bin probabilities with a Gaussian. It is the
prior of the domain market makers and the benchmark of the efficiency study.
"""
import datetime as dt
import math

import numpy as np
import pandas as pd

import cities
from common import DATA, read_gz
from forecast import probabilities as forecasts

HOURS = (-12, -6, 0, 6, 9, 12, 15)
LATENCY_HOURS = {'nbm': 1.25, 'gfs_mos': 3.75}  # Typical bulletin availability after the model cycle.
BIAS_WINDOW = 21
MIN_HISTORY = 10
TAIL_WIDTH = 3.0  # Assumed width (F) of the open-ended bins when reading a median off the prices.


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
                          'hour': (hours[j] - info['start'].timestamp()) / 3600, 'probs': probs,
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


def public_brier(row, info):
    """Brier of Gaussian bins around the bias-corrected NBM forecast; tails absorb the open-ended mass."""
    sd = max(1.0, row['error_sd'] if pd.notna(row['error_sd']) else 2.5)
    p = np.array([forecasts.normal_cdf(hi, row['public'], sd) - forecasts.normal_cdf(lo, row['public'], sd)
                  for lo, hi in zip(info['lows'], info['highs'])])
    p[0] += forecasts.normal_cdf(info['lows'][0], row['public'], sd)
    p[-1] += 1 - forecasts.normal_cdf(info['highs'][-1], row['public'], sd)
    p = np.array(forecasts.floor_probabilities(list(p)))
    return float(np.mean((p - np.eye(6)[info['winner']]) ** 2)), p
