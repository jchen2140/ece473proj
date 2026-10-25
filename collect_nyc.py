"""Reproducible Central Park / Kalshi / GEFS pilot (public, read-only APIs)."""
import argparse
import csv
import datetime as dt
import email.utils
import http.client
import json
import math
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from zoneinfo import ZoneInfo

import config

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
UTC = dt.timezone.utc
EST = dt.timezone(dt.timedelta(hours=-5))  # NWS climate day: fixed standard time
NY = ZoneInfo('America/New_York')
API = 'https://external-api.kalshi.com/trade-api/v2'
S3 = 'https://noaa-gefs-pds.s3.amazonaws.com'
MEMBERS = ['gec00'] + [f'gep{i:02d}' for i in range(1, 31)]
DAY_CACHE_VERSION = 2
_GHCN_HIGHS = None


def ghcn_rounded_high(day):
    """Quality-screened Central Park GHCN high (rounded half up), or None if not on file."""
    global _GHCN_HIGHS
    if _GHCN_HIGHS is None:
        path = DATA / 'historical_tmax.csv'
        with path.open(newline='') as f:
            _GHCN_HIGHS = {r['date']: int(r['rounded_tmax_f']) for r in csv.DictReader(f)} if path.exists() else {}
    return _GHCN_HIGHS.get(day.isoformat())


class SkipDay(Exception):
    """A day that fails a strict per-day check; it is logged and left out, not fatal."""
    def __init__(self, reason, detail):
        super().__init__(f'{reason}: {detail}')
        self.reason, self.detail = reason, detail


def fetch(url, headers=None):
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'ECE473-NYC-research/1.0', **(headers or {})})
            with urllib.request.urlopen(req, timeout=60) as response:
                return response.read(), dict(response.headers), response.status
        except (OSError, http.client.HTTPException) as exc:  # URLError, socket.timeout, resets
            if isinstance(exc, urllib.error.HTTPError) and exc.code not in (429, 500, 502, 503, 504):
                raise
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)


def cached_json(name, url):
    path = DATA / 'raw' / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        body, _, _ = fetch(url)
        value = json.loads(body)
        path.write_text(json.dumps(value, indent=2))
        path.with_suffix('.source.json').write_text(json.dumps({'url': url, 'retrieved_at': dt.datetime.now(UTC).isoformat()}, indent=2))
    return json.loads(path.read_text())


def write_csv(name, rows):
    path = DATA / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def event_name(day):
    months = ('JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC')
    return f'KXHIGHNY-{day:%y}{months[day.month-1]}{day:%d}'


def get_markets(day):
    event = event_name(day)
    markets = {}
    # Query both storage tiers; a 404 or an empty response in one is not absence.
    for route in ('historical/markets', 'markets'):
        cursor = ''
        page = 0
        while True:
            params = {'event_ticker': event, 'limit': 1000}
            if cursor:
                params['cursor'] = cursor
            payload = cached_json(f'kalshi/{event}_{route.replace("/", "_")}_{page}.json', API + '/' + route + '?' + urllib.parse.urlencode(params))
            for market in payload['markets']:
                if market['event_ticker'] != event:
                    raise SkipDay('unexpected_event', f'{route} returned {market["event_ticker"]} for {event}')
                if market.get('result') in ('yes', 'no'):
                    markets[market['ticker']] = (market, route.startswith('historical'))
            cursor = payload.get('cursor', '')
            if not cursor:
                break
            page += 1
    if len(markets) != 6:
        raise SkipDay('wrong_bin_count', f'{event}: expected six settled bins, got {len(markets)}')
    for market, _ in markets.values():
        rules = market['rules_primary']
        if 'Central Park' not in rules or 'National Weather Service' not in rules:
            raise SkipDay('non_nws_rules', f'{event}: rules do not name Central Park and the NWS')
    return list(markets.values())


def candle_at(market, historical, cutoff):
    ticker = market['ticker']
    route = f'historical/markets/{ticker}/candlesticks' if historical else f'series/KXHIGHNY/markets/{ticker}/candlesticks'
    params = {'start_ts': int(cutoff.timestamp()) - 7200, 'end_ts': int(cutoff.timestamp()), 'period_interval': 60}
    payload = cached_json(f'kalshi/{ticker}_candles.json', API + '/' + route + '?' + urllib.parse.urlencode(params))
    valid = [c for c in payload['candlesticks'] if c['end_period_ts'] <= cutoff.timestamp()]
    if not valid:
        return None, None, None, None
    c = max(valid, key=lambda x: x['end_period_ts'])
    def price(field):
        value = c.get(field, {}).get('close_dollars', c.get(field, {}).get('close'))
        return float(value) if value is not None else None
    bid, ask = price('yes_bid'), price('yes_ask')
    # Dollar values in the current response schema: never divide by 100 blindly.
    if any(v is not None and not 0 <= v <= 1 for v in (bid, ask)):
        raise ValueError('Unexpected price units')
    midpoint = (bid + ask) / 2 if bid is not None and ask is not None and bid <= ask else None
    return bid, ask, midpoint, dt.datetime.fromtimestamp(c['end_period_ts'], UTC).isoformat()


def gefs_point(task):
    from eccodes import codes_new_from_message, codes_get, codes_grib_find_nearest, codes_release
    day, member, lead, latitude, longitude, cutoff = task
    run = dt.datetime.combine(day - dt.timedelta(days=1), dt.time(6), UTC)
    stem = f'{member}.t06z.pgrb2s.0p25.f{lead:03d}'
    url = f'{S3}/gefs.{run:%Y%m%d}/06/atmos/pgrb2sp25/{stem}'
    cache = DATA / 'raw' / 'gefs' / f'{run:%Y%m%d}06' / f'{stem}.t2m.grib2'
    cache.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = cache.with_suffix('.json')
    if not cache.exists() or not metadata_path.exists():
        index, _, _ = fetch(url + '.idx')
        lines = index.decode().splitlines()
        matches = [i for i, line in enumerate(lines) if ':TMP:2 m above ground:' in line]
        if len(matches) != 1:
            raise ValueError(f'Unexpected temperature fields: {url}')
        i = matches[0]
        start = int(lines[i].split(':')[1])
        end = int(lines[i+1].split(':')[1]) - 1
        body, headers, status = fetch(url, {'Range': f'bytes={start}-{end}'})
        if status != 206 or len(body) != end-start+1 or body[:4] != b'GRIB':
            raise ValueError(f'Invalid GRIB range response: {url}')
        modified = headers.get('Last-Modified') or headers.get('last-modified')
        metadata = {'url': url, 'index_url': url + '.idx', 'index_record': lines[i],
                    'byte_range': [start, end], 'last_modified': modified,
                    'retrieved_at': dt.datetime.now(UTC).isoformat()}
        cache.write_bytes(body)
        metadata_path.write_text(json.dumps(metadata, indent=2))
    metadata = json.loads(metadata_path.read_text())
    modified = email.utils.parsedate_to_datetime(metadata['last_modified'])
    if modified > cutoff:
        raise SkipDay('gefs_after_cutoff', f'Archive timestamp later than prediction cutoff: {url}')
    handle = codes_new_from_message(cache.read_bytes())
    try:
        if codes_get(handle, 'shortName') != '2t' or codes_get(handle, 'forecastTime') != lead:
            raise ValueError('Unexpected GRIB variable or lead')
        if codes_get(handle, 'dataDate') != int(run.strftime('%Y%m%d')) or codes_get(handle, 'dataTime') != 600:
            raise ValueError('Unexpected model initialization')
        # Use the geometric nearest grid point; save its distance for coastal-bias review.
        point = codes_grib_find_nearest(handle, latitude, longitude)[0]
        return {'target_date': day.isoformat(), 'model_run_utc': run.isoformat(), 'member': member,
                'forecast_hour': lead, 'valid_time_utc': (run + dt.timedelta(hours=lead)).isoformat(),
                'temperature_f': (point['value'] - 273.15)*9/5 + 32,
                'grid_latitude': point['lat'], 'grid_longitude': point['lon'],
                'distance_km': point['distance'], 'archive_last_modified': modified.isoformat()}
    finally:
        codes_release(handle)


def hits(market, value):
    kind = market['strike_type']
    if kind == 'less':
        return value < market['cap_strike']
    if kind == 'greater':
        return value > market['floor_strike']
    if kind == 'between':
        return market['floor_strike'] <= value <= market['cap_strike']
    raise ValueError(f'Unsupported strike type {kind}')


def collect_day(day, latitude, longitude, workers):
    """Build all rows for one target date, or raise SkipDay. Strict checks are unchanged."""
    previous = day - dt.timedelta(days=1)
    cutoff = dt.datetime.combine(previous, dt.time(12), NY).astimezone(UTC)
    pairs = get_markets(day)
    contracts, maxima, matched = [], [], []
    for market, historical in pairs:
        bid, ask, midpoint, candle_time = candle_at(market, historical, cutoff)
        contracts.append({'target_date': day.isoformat(), 'market_ticker': market['ticker'],
                          'event_ticker': market['event_ticker'], 'bin_label': market['yes_sub_title'],
                          'strike_type': market['strike_type'], 'lower_bound': market.get('floor_strike'),
                          'upper_bound': market.get('cap_strike'), 'outcome': int(market['result'] == 'yes'),
                          'kalshi_settlement_high_f': market['expiration_value'],
                          'prediction_time_utc': cutoff.isoformat(), 'market_bid': bid, 'market_ask': ask,
                          'market_midpoint': midpoint, 'market_candle_end_utc': candle_time,
                          'rules_primary': market['rules_primary']})
    # Kalshi left expiration_value blank on some 2025 events while still settling every bin.
    # Then use the GHCN Central Park high, accepted only if it lands in Kalshi's winning bin.
    # Some events leave the field blank on only some bins; the reported bins must agree.
    values = {market.get('expiration_value') for market, _ in pairs} - {None, ''}
    if not values:
        fallback = ghcn_rounded_high(day)
        if fallback is None:
            raise SkipDay('missing_settlement_high', f'{day}: Kalshi expiration_value is empty and no GHCN high')
        winners = [market for market, _ in pairs if market['result'] == 'yes']
        if len(winners) != 1 or not hits(winners[0], fallback):
            raise SkipDay('ghcn_fallback_inconsistent', f'{day}: GHCN high {fallback} outside Kalshi winning bin')
        pairs = [({**market, 'expiration_value': f'{fallback:.2f}'}, historical) for market, historical in pairs]
        source = 'ghcn_fallback'
    elif len(values) != 1:
        raise SkipDay('inconsistent_settlement_high', f'{day}: bins report different expiration_values {sorted(values)}')
    else:
        reported = values.pop()
        pairs = [({**market, 'expiration_value': reported}, historical) for market, historical in pairs]
        source = 'kalshi'
    for c in contracts:
        c['kalshi_settlement_high_f'] = pairs[0][0]['expiration_value']
        c['settlement_high_source'] = source
    if sum(c['outcome'] for c in contracts) != 1:
        raise SkipDay('outcome_count', f'{day}: expected exactly one YES bin')
    # 03Z before day start through 06Z after day end: permits interpolation at 05Z boundaries.
    tasks = [(day, m, lead, latitude, longitude, cutoff) for m in MEMBERS for lead in range(21, 49, 3)]
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            points = list(pool.map(gefs_point, tasks))
    except urllib.error.HTTPError as exc:
        raise SkipDay('gefs_missing', f'{day}: HTTP {exc.code} {exc.url}') from exc
    except ValueError as exc:
        raise SkipDay('gefs_invalid', f'{day}: {exc}') from exc
    for member in MEMBERS:
        series = sorted([p for p in points if p['member'] == member], key=lambda p: p['forecast_hour'])
        values = [p['temperature_f'] for p in series]
        # NWS day is 05Z..05Z. Interpolated edges plus eight internal 3-hourly samples.
        samples = [values[0]/3 + values[1]*2/3] + values[1:-1] + [values[-2]/3 + values[-1]*2/3]
        high = max(samples)
        maxima.append({'target_date': day.isoformat(), 'member': member,
                       'model_run_utc': series[0]['model_run_utc'], 'daily_high_approx_f': high,
                       'rounded_high_f': math.floor(high + 0.5),
                       'day_start_utc': dt.datetime.combine(day, dt.time(), EST).astimezone(UTC).isoformat(),
                       'day_end_utc': dt.datetime.combine(day+dt.timedelta(days=1), dt.time(), EST).astimezone(UTC).isoformat()})
    assert len(maxima) == 31
    for market, _ in pairs:
        base = next(c for c in contracts if c['market_ticker'] == market['ticker'])
        count = sum(hits(market, m['rounded_high_f']) for m in maxima)
        if hits(market, float(market['expiration_value'])) != bool(base['outcome']):
            raise SkipDay('outcome_inconsistent', f'{market["ticker"]}: outcome inconsistent with reported settlement high')
        matched.append({**base, 'station_id': 'KNYC', 'latitude': latitude, 'longitude': longitude,
                        'model_run_utc': maxima[0]['model_run_utc'], 'member_count': 31,
                        'members_in_bin': count, 'ensemble_probability': count/31,
                        'probability_method': 'uncalibrated_rounded_3hourly_max_with_interpolated_edges'})
    if sum(r['members_in_bin'] for r in matched) != 31:
        raise SkipDay('bins_not_partition', f'{day}: contract bins do not partition ensemble members')
    return {'version': DAY_CACHE_VERSION, 'contracts': contracts, 'points': points,
            'maxima': maxima, 'matched': matched}


def load_or_collect_day(day, latitude, longitude, workers):
    """Each finished day is saved on its own, so an interrupted run resumes at the next day."""
    path = DATA / 'days' / f'{day.isoformat()}.json'
    if path.exists():
        cached = json.loads(path.read_text())
        if cached.get('version') == DAY_CACHE_VERSION:
            return cached
    result = collect_day(day, latitude, longitude, workers)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start', default=config.WINDOW_START.isoformat())
    parser.add_argument('--days', type=int, default=config.window_days())
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    first = dt.date.fromisoformat(args.start)
    station = cached_json('station_KNYC.json', 'https://api.weather.gov/stations/KNYC')
    longitude, latitude = station['geometry']['coordinates']
    contracts, points, maxima, matched, skipped = [], [], [], [], []
    started = time.monotonic()
    for offset in range(args.days):
        day = first + dt.timedelta(days=offset)
        try:
            result = load_or_collect_day(day, latitude, longitude, args.workers)
        except SkipDay as exc:
            skipped.append({'target_date': day.isoformat(), 'event_ticker': event_name(day),
                            'reason': exc.reason, 'detail': exc.detail})
            print(f'{day}: SKIPPED ({exc.reason}) {exc.detail}', flush=True)
            continue
        except (urllib.error.URLError, TimeoutError, OSError, http.client.HTTPException) as exc:
            # Network failures are not cached, so a rerun retries these days.
            skipped.append({'target_date': day.isoformat(), 'event_ticker': event_name(day),
                            'reason': 'network_error', 'detail': f'{type(exc).__name__}: {exc}'})
            print(f'{day}: SKIPPED (network_error) {exc}', flush=True)
            continue
        contracts += result['contracts']
        points += result['points']
        maxima += result['maxima']
        matched += result['matched']
        print(f'{day}: {len(result["matched"])} contracts saved ({time.monotonic()-started:.0f}s elapsed)', flush=True)
    if not matched:
        raise SystemExit('No days collected')
    write_csv('markets.csv', contracts)
    write_csv('ensemble_temperatures.csv', points)
    write_csv('ensemble_daily_maxima.csv', maxima)
    write_csv('matched.csv', matched)
    with (DATA/'skipped_days.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['target_date', 'event_ticker', 'reason', 'detail'], lineterminator='\n')
        writer.writeheader()
        writer.writerows(skipped)
    days = len({r['target_date'] for r in matched})
    report = {'requested_start': first.isoformat(), 'requested_days': args.days,
              'days': days, 'skipped_days': len(skipped), 'contracts': len(matched), 'ensemble_members_per_day': 31,
              'temperature_points': len(points), 'missing_market_midpoints': sum(r['market_midpoint'] is None for r in matched),
              'ensemble_brier_mean_per_binary_contract': sum((r['ensemble_probability']-r['outcome'])**2 for r in matched)/len(matched),
              'limitations': ['Descriptive all-days score; out-of-sample comparisons are in data/results/.',
                             'GEFS 0.25-degree nearest grid point is not a station measurement.',
                             'Daily maxima approximate continuous extrema using three-hourly temperatures and interpolated 05Z edges.',
                             'Round-half-up is an explicit modeling convention, not a verified model of NWS instrument processing.',
                             'Actual high is Kalshi expiration_value; original NWS climate reports have not been independently downloaded.',
                             'Raw ensemble probabilities have no bias or spread correction.',
                             'S3 Last-Modified checks support availability timing; they do not prove exact historical dissemination latency.']}
    complete = [r for r in matched if r['market_midpoint'] is not None]
    if complete:
        report['market_brier_mean_per_binary_contract'] = sum((r['market_midpoint']-r['outcome'])**2 for r in complete)/len(complete)
    (DATA/'validation.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f'Finished in {time.monotonic()-started:.0f}s; {days} days collected, {len(skipped)} skipped.')


if __name__ == '__main__':
    main()
