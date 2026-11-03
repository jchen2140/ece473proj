"""Kalshi quotes, NWS MOS forecasts and ASOS observations for every city in cities.py.

Per city, written to data/cities/<code>/ (raw responses cached under data/raw/cities/, gitignored):
- markets.csv: settled six-bin events in the window whose rules name the NWS Climatological Report.
- candles.csv.gz: hourly bid/ask/volume over each contract's whole life.
- nbm.csv.gz, gfs_mos.csv.gz: every archived NBM (NBS) and GFS MOS (MAV) run: the daily max for
  each target day, its spread (NBM only), and the 3-hourly temperature forecast.
- asos.csv.gz: routine and special observations at the settlement station.
- validation.json: settlement high versus the ASOS maximum for the same climate day.

Public, read-only and resumable. Kalshi calls are throttled to stay under the public rate limit.
"""
import argparse
import csv
import datetime as dt
import gzip
import io
import json
import re
import threading
import time
import urllib.error
import urllib.parse
from concurrent.futures import ThreadPoolExecutor

import cities
from collect.features import IEM_PAUSE, MOS, ASOS, iem_text, month_starts, number
from common import DATA, UTC
from collect.nyc import API, fetch

OUT = DATA / 'cities'
RAW = DATA / 'raw' / 'cities'
KALSHI_INTERVAL = 0.35  # Seconds between Kalshi requests across all threads; faster rates draw 429s.
MONTHS = {m: i + 1 for i, m in enumerate(('JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC'))}
_lock = threading.Lock()
_last = [0.0]


def kalshi(route, params):
    for attempt in range(8):
        with _lock:
            wait = _last[0] + KALSHI_INTERVAL - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            _last[0] = time.monotonic()
        try:
            body, _, _ = fetch(API + '/' + route + '?' + urllib.parse.urlencode(params))
            return json.loads(body)
        except urllib.error.HTTPError as exc:
            if exc.code != 429 or attempt == 7:
                raise
            with _lock:  # Back off every thread, not just this one.
                _last[0] = time.monotonic() + 20 * (attempt + 1)


def write_gz(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, 'wt', newline='', encoding='utf-8', compresslevel=9) as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def event_date(event_ticker):
    match = re.search(r'-(\d{2})([A-Z]{3})(\d{2})$', event_ticker)
    return dt.date(2000 + int(match[1]), MONTHS[match[2]], int(match[3])) if match else None


# ---------------------------------------------------------------- Kalshi

def series_markets(city):
    """Every settled market in the series from both storage tiers, cached per city."""
    path = RAW / city.code / 'markets.json'
    if path.exists():
        return json.loads(path.read_text(encoding='utf-8'))
    markets = {}
    for route in ('historical/markets', 'markets'):
        cursor = ''
        while True:
            params = {'series_ticker': city.series, 'limit': 1000, **({'status': 'settled'} if route == 'markets' else {})}
            if cursor:
                params['cursor'] = cursor
            payload = kalshi(route, params)
            for m in payload.get('markets') or []:
                markets[m['ticker']] = {**m, 'tier': 'historical' if route.startswith('historical') else 'live'}
            cursor = payload.get('cursor') or ''
            if not cursor or not payload.get('markets'):
                break
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(list(markets.values())), encoding='utf-8')
    return list(markets.values())


def window_events(city, markets):
    """Six-bin settled events in the window under NWS rules; anything else is counted and dropped."""
    events, dropped = {}, {}
    for m in markets:
        day = event_date(m['event_ticker'])
        if day and cities.WINDOW_START <= day <= cities.WINDOW_END and m.get('result') in ('yes', 'no'):
            events.setdefault(day, []).append(m)
    rows = []
    for day, group in sorted(events.items()):
        if len(group) != 6:
            dropped[str(day)] = f'{len(group)} bins'
            continue
        if not all('National Weather Service' in m['rules_primary'] for m in group):
            dropped[str(day)] = 'not NWS rules'
            continue
        for m in sorted(group, key=lambda m: m['ticker']):
            rows.append({'city': city.code, 'target_date': day.isoformat(), 'market_ticker': m['ticker'],
                         'strike_type': m['strike_type'], 'lower_bound': m.get('floor_strike') if m.get('floor_strike') is not None else '',
                         'upper_bound': m.get('cap_strike') if m.get('cap_strike') is not None else '',
                         'outcome': int(m['result'] == 'yes'), 'settlement_high_f': m.get('expiration_value') or '',
                         'open_time': m['open_time'], 'close_time': m['close_time'], 'tier': m['tier']})
    return rows, dropped


def candles(job):
    city, row = job
    path = RAW / city.code / 'candles' / f"{row['market_ticker']}.json"
    if path.exists():
        return row, json.loads(path.read_text(encoding='utf-8'))
    ticker = row['market_ticker']
    route = (f'historical/markets/{ticker}/candlesticks' if row['tier'] == 'historical'
             else f'series/{city.series}/markets/{ticker}/candlesticks')
    start = int(dt.datetime.fromisoformat(row['open_time'].replace('Z', '+00:00')).timestamp()) - 3600
    end = int(dt.datetime.fromisoformat(row['close_time'].replace('Z', '+00:00')).timestamp()) + 3600
    value = kalshi(route, {'start_ts': start, 'end_ts': end, 'period_interval': 60}).get('candlesticks') or []
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix('.part')
    partial.write_text(json.dumps(value), encoding='utf-8')
    partial.replace(path)
    return row, value


def close_value(candle, field):
    part = candle.get(field) or {}
    value = part.get('close_dollars', part.get('close'))
    return value if value is not None else ''


# ---------------------------------------------------------------- IEM

def mos_rows(city, model):
    """Every archived run: per target day, the max-temperature forecast; plus 3-hourly temperatures."""
    out = []
    start, end = cities.WINDOW_START - dt.timedelta(days=4), cities.WINDOW_END + dt.timedelta(days=1)
    for month in month_starts(start, end):
        stop = (month + dt.timedelta(days=32)).replace(day=1)
        url = f'{MOS}?station={city.icao}&model={model}&sts={month:%Y-%m-%d}T00:00Z&ets={stop:%Y-%m-%d}T00:00Z&format=csv'
        text = iem_text(f'cities/{city.code}/mos/{model}_{month:%Y%m}.csv', url)
        max_column = 'txn' if model == 'NBS' else 'n_x'
        for r in csv.DictReader(io.StringIO(text)):
            ftime = r['ftime']
            # The daytime maximum is reported on the 00 UTC row for every US time zone; it belongs to the previous date.
            is_max = ftime.endswith('00:00:00') and number(r.get(max_column)) is not None
            out.append({'runtime': r['runtime'], 'ftime': ftime, 'tmp_f': number(r.get('tmp')),
                        'max_f': number(r.get(max_column)) if is_max else None,
                        'max_sd_f': number(r.get('xnd')) if is_max and model == 'NBS' else None})
    return [{k: '' if v is None else v for k, v in r.items()} for r in out]


def asos_rows(city):
    a, b = cities.WINDOW_START - dt.timedelta(days=3), cities.WINDOW_END + dt.timedelta(days=2)
    url = (f'{ASOS}?station={city.asos}&data=tmpf&tz=Etc/UTC&format=onlycomma&missing=M&report_type=3&report_type=4'
           f'&year1={a.year}&month1={a.month}&day1={a.day}&year2={b.year}&month2={b.month}&day2={b.day}')
    text = iem_text(f'cities/{city.code}/asos_{a:%Y%m%d}_{b:%Y%m%d}.csv', url)
    return [{'valid_utc': r['valid'], 'tmpf': r['tmpf']} for r in csv.DictReader(io.StringIO(text)) if number(r['tmpf']) is not None]


def validate(city, markets, obs):
    """Settlement high versus the station's ASOS maximum over the same climate day (standard time)."""
    highs = {r['target_date']: float(r['settlement_high_f']) for r in markets if r['settlement_high_f'] != ''}
    times = [(dt.datetime.fromisoformat(o['valid_utc']).replace(tzinfo=UTC), float(o['tmpf'])) for o in obs]
    gaps = []
    for day, high in highs.items():
        start = cities.day_start_utc(city, dt.date.fromisoformat(day))
        values = [t for when, t in times if start <= when < start + dt.timedelta(days=1)]
        if values:
            gaps.append(high - round(max(values)))
    return {'days': len(gaps), 'within_1f': sum(abs(g) <= 1 for g in gaps) / max(1, len(gaps)),
            'mean_gap_f': sum(gaps) / max(1, len(gaps))}


def collect(city, workers):
    out = OUT / city.code
    out.mkdir(parents=True, exist_ok=True)
    markets, dropped = window_events(city, series_markets(city))
    with (out / 'markets.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(markets[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(markets)
    rows = []
    with ThreadPoolExecutor(workers) as pool:
        for market, values in pool.map(candles, [(city, m) for m in markets]):
            for c in values:
                rows.append({'target_date': market['target_date'], 'market_ticker': market['market_ticker'],
                             'end_period_ts': c['end_period_ts'], 'yes_bid': close_value(c, 'yes_bid'),
                             'yes_ask': close_value(c, 'yes_ask'), 'volume': c.get('volume_fp', c.get('volume', ''))})
    write_gz(out / 'candles.csv.gz', rows)
    write_gz(out / 'nbm.csv.gz', mos_rows(city, 'NBS'))
    write_gz(out / 'gfs_mos.csv.gz', mos_rows(city, 'GFS'))
    obs = asos_rows(city)
    write_gz(out / 'asos.csv.gz', obs)
    report = {'city': city.code, 'series': city.series, 'days': len(markets) // 6, 'dropped_days': dropped,
              'candles': len(rows), 'asos_check': validate(city, markets, obs)}
    (out / 'validation.json').write_text(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cities', nargs='*', default=[c.code for c in cities.CITIES])
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    for code in args.cities:
        report = collect(cities.BY_CODE[code], args.workers)
        check = report['asos_check']
        print(f"{code}: {report['days']} days, {report['candles']:,} candles, {len(report['dropped_days'])} dropped; "
              f"settlement within 1F of ASOS max on {check['within_1f']:.0%} of {check['days']} days", flush=True)


if __name__ == '__main__':
    main()
