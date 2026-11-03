"""Full-life Kalshi trades and hourly candles for every contract in data/markets.csv.

collect/nyc.py only keeps a two-hour candle snapshot before the noon cutoff. The backtest needs
every real taker trade and the quote path, so this pulls them per contract (public, read-only,
resumable). Raw per-contract files go to data/raw/kalshi_flow/ (gitignored); the assembled,
compressed tables are data/flow/trades.csv.gz and data/flow/candles.csv.gz.
"""
import argparse
import csv
import datetime as dt
import gzip
import json
import time
import urllib.error
import urllib.parse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from common import DATA, UTC
from collect.nyc import API, fetch

RAW = DATA / 'raw' / 'kalshi_flow'
OUT = DATA / 'flow'
TRADE_FIELDS = ['target_date', 'market_ticker', 'trade_id', 'created_time', 'taker_side', 'count',
                'yes_price', 'no_price', 'is_block_trade']
CANDLE_FIELDS = ['target_date', 'market_ticker', 'end_period_ts', 'yes_bid', 'yes_ask',
                 'price_close', 'volume', 'open_interest']
TIME_FORMATS = ('%Y-%m-%dT%H:%M:%SZ', '%Y-%m-%dT%H:%M:%S.%fZ')


def get_json(route, params):
    body, _, _ = fetch(API + '/' + route + '?' + urllib.parse.urlencode(params))
    return json.loads(body)


def parse_time(text):
    for fmt in TIME_FORMATS:
        try:
            return dt.datetime.strptime(text, fmt).replace(tzinfo=UTC)
        except ValueError:
            pass
    raise ValueError(f'Unexpected Kalshi time {text}')


def market_metadata():
    """ticker -> open/close times and tier, read from the market responses already cached."""
    out = {}
    for tier, pattern in (('historical', '*_historical_markets_*.json'), ('live', '*_markets_*.json')):
        for path in sorted((DATA / 'raw' / 'kalshi').glob(pattern)):
            if path.name.endswith('.source.json') or (tier == 'live' and '_historical_' in path.name):
                continue
            for market in json.loads(path.read_text(encoding='utf-8'))['markets']:
                if market.get('result') in ('yes', 'no'):
                    out[market['ticker']] = {'open': market['open_time'], 'close': market['close_time'], 'tier': tier}
    return out


def fetch_trades(ticker):
    """Trades from both storage tiers, deduplicated; a contract can straddle the cutoff."""
    trades = {}
    for route in ('historical/trades', 'markets/trades'):
        cursor = ''
        while True:
            params = {'ticker': ticker, 'limit': 1000}
            if cursor:
                params['cursor'] = cursor
            try:
                payload = get_json(route, params)
            except urllib.error.HTTPError as exc:
                if exc.code == 404:
                    break
                raise
            for trade in payload.get('trades') or []:
                trades[trade['trade_id']] = trade
            cursor = payload.get('cursor') or ''
            if not cursor:
                break
    return sorted(trades.values(), key=lambda t: t['created_time'])


def fetch_candles(ticker, meta):
    series = ticker.split('-')[0]
    route = (f'historical/markets/{ticker}/candlesticks' if meta['tier'] == 'historical'
             else f'series/{series}/markets/{ticker}/candlesticks')
    start = int(parse_time(meta['open']).timestamp()) - 3600
    end = int(parse_time(meta['close']).timestamp()) + 3600
    return get_json(route, {'start_ts': start, 'end_ts': end, 'period_interval': 60}).get('candlesticks') or []


def collect_contract(job):
    """Write one contract's raw file atomically, so an interrupted run resumes cleanly."""
    ticker, meta = job
    path = RAW / f'{ticker}.json'
    if path.exists():
        return ticker, 'cached'
    value = {'ticker': ticker, 'meta': meta, 'retrieved_at': dt.datetime.now(UTC).isoformat(),
             'trades': fetch_trades(ticker), 'candles': fetch_candles(ticker, meta)}
    partial = path.with_suffix('.partial')
    partial.write_text(json.dumps(value), encoding='utf-8')
    partial.replace(path)
    return ticker, 'fetched'


def close_value(candle, field):
    value = (candle.get(field) or {}).get('close')
    return value if value is not None else ''


def assemble(dates):
    OUT.mkdir(parents=True, exist_ok=True)
    trades_written = candles_written = missing = 0
    with gzip.open(OUT / 'trades.csv.gz', 'wt', newline='', encoding='utf-8', compresslevel=9) as tf, \
            gzip.open(OUT / 'candles.csv.gz', 'wt', newline='', encoding='utf-8', compresslevel=9) as cf:
        trades_out = csv.DictWriter(tf, TRADE_FIELDS, lineterminator='\n')
        candles_out = csv.DictWriter(cf, CANDLE_FIELDS, lineterminator='\n')
        trades_out.writeheader()
        candles_out.writeheader()
        for ticker, day in sorted(dates.items(), key=lambda kv: (kv[1], kv[0])):
            path = RAW / f'{ticker}.json'
            if not path.exists():
                missing += 1
                continue
            value = json.loads(path.read_text(encoding='utf-8'))
            for t in value['trades']:
                trades_out.writerow({'target_date': day, 'market_ticker': ticker, 'trade_id': t['trade_id'],
                                     'created_time': t['created_time'], 'taker_side': t['taker_side'],
                                     'count': t['count_fp'], 'yes_price': t['yes_price_dollars'],
                                     'no_price': t['no_price_dollars'], 'is_block_trade': int(bool(t.get('is_block_trade')))})
                trades_written += 1
            for c in value['candles']:
                candles_out.writerow({'target_date': day, 'market_ticker': ticker, 'end_period_ts': c['end_period_ts'],
                                      'yes_bid': close_value(c, 'yes_bid'), 'yes_ask': close_value(c, 'yes_ask'),
                                      'price_close': close_value(c, 'price'), 'volume': c.get('volume', ''),
                                      'open_interest': c.get('open_interest', '')})
                candles_written += 1
    return {'contracts_requested': len(dates), 'contracts_missing': missing, 'trades': trades_written, 'candles': candles_written}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--start', help='First target date (default: all of data/markets.csv)')
    parser.add_argument('--days', type=int)
    args = parser.parse_args()
    with (DATA / 'markets.csv').open(newline='', encoding='utf-8') as f:
        dates = {r['market_ticker']: r['target_date'] for r in csv.DictReader(f)}
    if args.start:
        first = dt.date.fromisoformat(args.start)
        keep = {(first + dt.timedelta(days=i)).isoformat() for i in range(args.days or 1)}
        dates = {t: d for t, d in dates.items() if d in keep}
    metadata = market_metadata()
    absent = sorted(set(dates) - set(metadata))
    if absent:
        raise SystemExit(f'{len(absent)} contracts have no cached market metadata, e.g. {absent[:3]}')
    RAW.mkdir(parents=True, exist_ok=True)
    jobs = [(t, metadata[t]) for t in sorted(dates)]
    started = time.monotonic()
    tally, failures = Counter(), []

    def run(job):
        try:
            return collect_contract(job)
        except (OSError, ValueError, KeyError) as exc:
            return job[0], f'failed: {type(exc).__name__}: {exc}'

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, (ticker, status) in enumerate(pool.map(run, jobs), 1):
            tally[status.split(':')[0]] += 1
            if status.startswith('failed'):
                failures.append({'ticker': ticker, 'status': status})
            if i % 100 == 0 or i == len(jobs):
                print(f'{i}/{len(jobs)} {dict(tally)} {time.monotonic()-started:.0f}s', flush=True)
    report = {**assemble(dates), 'run': dict(tally), 'failures': failures,
              'finished_at': dt.datetime.now(UTC).isoformat()}
    (OUT / 'collection_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'failures'}, indent=2), f'failures={len(failures)}')


if __name__ == '__main__':
    main()
