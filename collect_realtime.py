"""Minute-level station temperatures and second-level Kalshi trades for the real-time latency study.

For each city in REALTIME_CITIES, over the cities.py window:
- 1-minute ASOS temperatures at the settlement station (Iowa Environmental Mesonet archive of the
  NCEI one-minute data). The public 5-minute feed and the hourly METAR are samples of this series.
- Every Kalshi trade in the city's contracts, from both storage tiers. NYC reuses data/flow/trades.csv.gz.
- The routine hourly METAR times, from data/cities/<code>/asos.csv.gz (already collected).

Large raw files go to data/raw/realtime/ (gitignored); realtime.py writes the committed results.
Run after collect_cities.py.
"""
import argparse
import csv
import datetime as dt
import gzip
import io
import json
from concurrent.futures import ThreadPoolExecutor

import cities
from collect_cities import kalshi
from collect_features import iem_text, month_starts
from collect_nyc import DATA

RAW = DATA / 'raw' / 'realtime'
ONE_MINUTE = 'https://mesonet.agron.iastate.edu/cgi-bin/request/asos1min.py'
REALTIME_CITIES = ['LAX', 'NYC', 'CHI', 'AUS', 'MSP', 'DCA']  # Busiest to thinnest by daily volume.


def one_minute(city):
    rows = []
    start, end = cities.WINDOW_START - dt.timedelta(days=1), cities.WINDOW_END + dt.timedelta(days=2)
    for month in month_starts(start, end):
        a = max(start, month)
        b = min(end, (month + dt.timedelta(days=32)).replace(day=1))
        url = (f'{ONE_MINUTE}?station={city.asos}&vars=tmpf&sts={a:%Y-%m-%d}T00:00Z&ets={b:%Y-%m-%d}T00:00Z'
               '&sample=1min&what=download&tz=UTC')
        text = iem_text(f'realtime/{city.code}/asos1min_{a:%Y%m%d}.csv', url)
        for r in csv.DictReader(io.StringIO(text)):
            if r['tmpf'] not in ('', 'M'):
                rows.append({'valid_utc': r['valid(UTC)'], 'tmpf': r['tmpf']})
    return rows


def ticker_trades(job):
    """All trades for one contract, cached as gzip JSON so an interrupted run resumes."""
    code, ticker = job
    path = RAW / code / 'trades' / f'{ticker}.json.gz'
    if path.exists():
        with gzip.open(path, 'rt', encoding='utf-8') as f:
            return ticker, json.load(f)
    trades = {}
    for route in ('historical/trades', 'markets/trades'):
        cursor = ''
        while True:
            params = {'ticker': ticker, 'limit': 1000, **({'cursor': cursor} if cursor else {})}
            try:
                payload = kalshi(route, params)
            except Exception as exc:  # A tier with no record of the ticker answers 404.
                if '404' in str(exc):
                    break
                raise
            for t in payload.get('trades') or []:
                trades[t['trade_id']] = [t['created_time'], t['taker_side'], t.get('count_fp', t.get('count')),
                                         t.get('yes_price_dollars', t.get('yes_price'))]
            cursor = payload.get('cursor') or ''
            if not cursor or not payload.get('trades'):
                break
    value = sorted(trades.values())
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix('.part')
    with gzip.open(partial, 'wt', encoding='utf-8') as f:
        json.dump(value, f)
    partial.replace(path)
    return ticker, value


def write_rows(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, 'wt', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(fields)
        writer.writerows(rows)


def collect(code, workers):
    city = cities.BY_CODE[code]
    with (DATA / 'cities' / code / 'markets.csv').open(newline='', encoding='utf-8') as f:
        tickers = [r['market_ticker'] for r in csv.DictReader(f)]
    minute = one_minute(city)
    write_rows(RAW / code / 'asos1min.csv.gz', [(r['valid_utc'], r['tmpf']) for r in minute], ['valid_utc', 'tmpf'])
    out = []
    if code == 'NYC':  # Already collected for the backtest.
        wanted = set(tickers)
        with gzip.open(DATA / 'flow' / 'trades.csv.gz', 'rt', newline='', encoding='utf-8') as f:
            for t in csv.DictReader(f):
                if t['market_ticker'] in wanted:
                    out.append((t['market_ticker'], t['created_time'], t['taker_side'], t['count'], t['yes_price']))
    else:
        with ThreadPoolExecutor(workers) as pool:
            for ticker, trades in pool.map(ticker_trades, [(code, t) for t in tickers]):
                out.extend((ticker, *t) for t in trades)
    write_rows(RAW / code / 'trades.csv.gz', out, ['market_ticker', 'created_time', 'taker_side', 'count', 'yes_price'])
    return len(minute), len(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cities', nargs='*', default=REALTIME_CITIES)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    for code in args.cities:
        minutes, trades = collect(code, args.workers)
        print(f'{code}: {minutes:,} one-minute temperatures, {trades:,} trades', flush=True)


if __name__ == '__main__':
    main()
