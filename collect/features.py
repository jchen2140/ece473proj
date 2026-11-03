"""Extra forecast sources and observations for the ML model, all knowable at the noon cutoff.

One row per target day D from HISTORY_START to the end of the study window, written to
data/features/daily_features.csv. The cutoff is noon America/New_York on D-1, as in collect/nyc.py.

- ECMWF IFS HRES, 00 UTC run on D-1 (0.25 degree open data, published about 08 UTC). Only the
  needed GRIB messages are fetched by byte range: 3-hourly 2 m temperature over the climate day,
  and 2 m dewpoint, 850 hPa temperature and sea-level pressure at 18 UTC on D where published.
- NWS MOS from the Iowa Environmental Mesonet archive: NBM (NBS), GFS (MAV) and NAM (MET). The
  latest run with runtime + LATENCY <= cutoff is used, so every bulletin was public at the cutoff.
- ASOS observations at Central Park, LaGuardia, JFK and Newark up to the cutoff.
- The GHCN Central Park high is the target; the D-2 high (final by the cutoff) is a feature.

Raw responses are cached under data/raw/{ecmwf,mos,asos}/ (gitignored); reruns reuse them.
"""
import argparse
import bisect
import csv
import datetime as dt
import io
import json
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import config
from common import DATA, NY, UTC, write_csv
from collect.nyc import fetch

HISTORY_START = dt.date(2024, 3, 2)  # First 0.25-degree ECMWF open-data run is 2024-03-01.
LATITUDE, LONGITUDE = 40.78333, -73.96667
ECMWF = 'https://storage.googleapis.com/ecmwf-open-data'
# 2 m temperature every 3 hours, 06 UTC on D to 06 UTC on D+1 (the 05-05 UTC climate day). The 3-hour
# maximum (mx2t3) would be better but is only in open data from Nov 2024, so the history uses 2t throughout.
MAX_STEPS = range(30, 55, 3)
AFTERNOON_STEP = 42  # 18 UTC on D, early afternoon local time.
AFTERNOON_PARAMS = {('2d', ''): 'ecmwf_dewpoint_18z_f', ('t', '850'): 'ecmwf_t850_18z_f', ('msl', ''): 'ecmwf_mslp_18z_hpa'}
MOS = 'https://mesonet.agron.iastate.edu/cgi-bin/request/mos.py'
ASOS = 'https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py'
MOS_MODELS = {'NBS': 'nbm', 'GFS': 'mav', 'NAM': 'met'}
# Hours from model runtime until the text bulletin is reliably public.
LATENCY = {'NBS': 3, 'GFS': 6, 'NAM': 6}
ASOS_STATIONS = ('NYC', 'LGA', 'JFK', 'EWR')
CLOUD = {'CL': 0, 'CLR': 0, 'SKC': 0, 'FW': 1, 'FEW': 1, 'SC': 2, 'SCT': 2, 'BK': 3, 'BKN': 3, 'OV': 4, 'OVC': 4, 'VV': 4}
IEM_PAUSE = 3.0  # IEM answers "Too many requests" to rapid calls.
SHORT_NAMES = {'max_2t': 'mx2t3'}  # Some ecCodes builds name the 3-hour maximum differently.
GRIB_LOCK = threading.Lock()


def kelvin_to_f(k):
    return (k - 273.15) * 9 / 5 + 32


def cutoff_for(day):
    """Noon New York time on D-1, in UTC."""
    return dt.datetime.combine(day - dt.timedelta(days=1), dt.time(12), NY).astimezone(UTC)


def days_between(start, end):
    return [start + dt.timedelta(days=i) for i in range((end - start).days + 1)]


# ---------------------------------------------------------------- ECMWF

def grib_point(body):
    from eccodes import codes_new_from_message, codes_get, codes_grib_find_nearest, codes_release
    with GRIB_LOCK:  # ecCodes' in-memory definitions fail under concurrent first use on Windows.
        handle = codes_new_from_message(body)
        try:
            point = codes_grib_find_nearest(handle, LATITUDE, LONGITUDE)[0]
            name = codes_get(handle, 'shortName')
            return SHORT_NAMES.get(name, name), point['value'], point['lat'], point['lon']
        finally:
            codes_release(handle)


def ecmwf_day(day):
    """Point values from the D-1 00 UTC run, cached as a small JSON per target day."""
    cache = DATA / 'raw' / 'ecmwf' / f'{day:%Y%m%d}.json'
    if cache.exists():
        return json.loads(cache.read_text())
    run = day - dt.timedelta(days=1)
    wanted = {step: [('2t', '')] for step in MAX_STEPS}
    wanted[AFTERNOON_STEP] = wanted[AFTERNOON_STEP] + list(AFTERNOON_PARAMS)
    out = {'target_date': day.isoformat(), 'ecmwf_run_utc': f'{run}T00:00:00+00:00'}
    maxima = []
    for step, params in wanted.items():
        url = f'{ECMWF}/{run:%Y%m%d}/00z/ifs/0p25/oper/{run:%Y%m%d}000000-{step}h-oper-fc'
        try:
            index, _, _ = fetch(url + '.index')
        except Exception as exc:  # A missing run is recorded, not fatal.
            out['ecmwf_error'] = f'{step}h: {exc}'
            return out
        records = [json.loads(line) for line in index.decode().splitlines() if line.strip()]
        for param, level in params:
            match = [r for r in records if r['param'] == param and r.get('levelist', '') == level]
            if not match and param != '2t':
                continue  # Early open data (before IFS 49r1, Nov 2024) lacks some afternoon fields.
            if len(match) != 1:
                raise ValueError(f'{url}: expected one {param}{level}, found {len(match)}')
            start, length = match[0]['_offset'], match[0]['_length']
            body, _, status = fetch(url + '.grib2', {'Range': f'bytes={start}-{start + length - 1}'})
            if status != 206 or body[:4] != b'GRIB':
                raise ValueError(f'Invalid GRIB range response: {url}')
            name, value, lat, lon = grib_point(body)
            if name != param:
                raise ValueError(f'{url}: decoded {name}, expected {param}')
            out['ecmwf_grid'] = f'{lat:.2f},{lon:.2f}'
            if param == '2t':
                maxima.append(kelvin_to_f(value))
            elif param == 'msl':
                out[AFTERNOON_PARAMS[param, level]] = value / 100
            else:
                out[AFTERNOON_PARAMS[param, level]] = kelvin_to_f(value)
    out['ecmwf_max_f'] = max(maxima)
    out['ecmwf_max_step_hour'] = list(MAX_STEPS)[maxima.index(max(maxima))]
    cache.parent.mkdir(parents=True, exist_ok=True)
    partial = cache.with_suffix('.part')  # Rename after writing, so an interrupted run leaves no truncated cache.
    partial.write_text(json.dumps(out, indent=2))
    partial.replace(cache)
    return out


# ---------------------------------------------------------------- IEM (MOS and ASOS)

def iem_text(name, url):
    path = DATA / 'raw' / name
    if path.exists():
        return path.read_text(encoding='utf-8')
    for attempt in range(6):
        body, _, _ = fetch(url)
        text = body.decode('utf-8')
        time.sleep(IEM_PAUSE)
        if not text.startswith('Too many requests'):
            break
        time.sleep(30 * (attempt + 1))
    else:
        raise RuntimeError(f'IEM kept refusing {url}')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')
    return text


def month_starts(start, end):
    month = start.replace(day=1)
    while month <= end:
        yield month
        month = (month + dt.timedelta(days=32)).replace(day=1)


def mos_runs(model, start, end):
    """runtime -> {ftime: row} for KNYC, fetched one month at a time."""
    runs = {}
    for month in month_starts(start, end):
        stop = (month + dt.timedelta(days=32)).replace(day=1)
        url = f'{MOS}?station=KNYC&model={model}&sts={month:%Y-%m-%d}T00:00Z&ets={stop:%Y-%m-%d}T00:00Z&format=csv'
        text = iem_text(f'mos/{model}_{month:%Y%m}.csv', url)
        for row in csv.DictReader(io.StringIO(text)):
            runtime = dt.datetime.fromisoformat(row['runtime']).replace(tzinfo=UTC)
            ftime = dt.datetime.fromisoformat(row['ftime']).replace(tzinfo=UTC)
            runs.setdefault(runtime, {})[ftime] = row
    return runs


def number(value):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return None if v <= -98 else v  # MOS uses -88/-99 as missing codes.


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def mos_features(model, runs, day):
    """Max temperature and afternoon conditions from the latest bulletin public at the cutoff."""
    prefix = MOS_MODELS[model]
    latest = cutoff_for(day) - dt.timedelta(hours=LATENCY[model])
    end = dt.datetime.combine(day + dt.timedelta(days=1), dt.time(0), UTC)
    candidates = [t for t in runs if t <= latest and end in runs[t]]
    if not candidates:
        return {}
    runtime = max(candidates)
    rows = runs[runtime]
    climate_day = [rows[t] for t in sorted(rows) if end - dt.timedelta(hours=19) < t <= end + dt.timedelta(hours=5)]
    afternoon = rows.get(end - dt.timedelta(hours=6), {})
    max_column = 'txn' if model == 'NBS' else 'n_x'
    out = {f'{prefix}_runtime_utc': runtime.isoformat(),
           f'{prefix}_lead_hours': (end - runtime).total_seconds() / 3600,
           f'{prefix}_max_f': number(rows[end].get(max_column)),
           f'{prefix}_hourly_max_f': max((v for v in (number(r.get('tmp')) for r in climate_day) if v is not None), default=None),
           f'{prefix}_dewpoint_18z_f': number(afternoon.get('dpt')),
           f'{prefix}_wind_kt_18z': number(afternoon.get('wsp')),
           f'{prefix}_wind_dir_18z': number(afternoon.get('wdr'))}
    if model == 'NBS':
        out.update({'nbm_max_sd_f': number(rows[end].get('xnd')),
                    'nbm_sky_pct': mean(number(r.get('sky')) for r in climate_day),
                    'nbm_precip_prob_12h': number(rows[end].get('p12')),
                    'nbm_solar_18z': number(afternoon.get('sol'))})
    else:
        out[f'{prefix}_cloud_18z'] = CLOUD.get((afternoon.get('cld') or '').strip())
        out[f'{prefix}_precip_prob_12h'] = number(rows[end].get('p12'))
    return out


def asos_obs(station, start, end):
    """Sorted (time, row) observations, fetched one year at a time."""
    obs = []
    for year in range(start.year, end.year + 1):
        a, b = max(start, dt.date(year, 1, 1)), min(end, dt.date(year, 12, 31))
        url = (f'{ASOS}?station={station}&data=tmpf&data=dwpf&data=sknt&data=drct&data=mslp&data=skyc1&data=skyc2'
               f'&tz=Etc/UTC&format=onlycomma&missing=M&report_type=3&report_type=4'
               f'&year1={a.year}&month1={a.month}&day1={a.day}&year2={b.year}&month2={b.month}&day2={b.day}')
        text = iem_text(f'asos/{station}_{year}_{a:%m%d}_{b:%m%d}.csv', url)
        for row in csv.DictReader(io.StringIO(text)):
            obs.append((dt.datetime.fromisoformat(row['valid']).replace(tzinfo=UTC), row))
    obs.sort(key=lambda o: o[0])
    return obs


def last_value(obs, times, when, field, window_hours=3):
    """Latest non-missing value at or before `when`, within the window."""
    i =bisect.bisect_right(times, when)
    while i > 0:
        i -= 1
        t, row = obs[i]
        if when - t > dt.timedelta(hours=window_hours):
            return None
        v = number(row.get(field))
        if v is not None:
            return v
    return None


def asos_features(station, obs, times, day):
    cutoff = cutoff_for(day)
    key = station.lower()
    temp = last_value(obs, times, cutoff, 'tmpf')
    out = {f'{key}_temp_f': temp}
    if station != 'NYC':
        return out
    # Climate day of D-1 starts at local standard midnight = 05 UTC.
    since = dt.datetime.combine(day - dt.timedelta(days=1), dt.time(5), UTC)
    so_far = [number(r['tmpf']) for t, r in obs[bisect.bisect_left(times, since):bisect.bisect_right(times, cutoff)]]
    so_far = [v for v in so_far if v is not None]
    direction, speed = last_value(obs, times, cutoff, 'drct'), last_value(obs, times, cutoff, 'sknt')
    pressure = last_value(obs, times, cutoff, 'mslp')
    pressure_before = last_value(obs, times, cutoff - dt.timedelta(hours=12), 'mslp')
    temp_before = last_value(obs, times, cutoff - dt.timedelta(hours=24), 'tmpf')
    out.update({'obs_dewpoint_f': last_value(obs, times, cutoff, 'dwpf'),
                'obs_wind_u_kt': None if direction is None or speed is None else -speed * math.sin(math.radians(direction)),
                'obs_wind_v_kt': None if direction is None or speed is None else -speed * math.cos(math.radians(direction)),
                'obs_mslp_hpa': pressure,
                'obs_mslp_change_12h': None if pressure is None or pressure_before is None else pressure - pressure_before,
                'obs_temp_change_24h': None if temp is None or temp_before is None else temp - temp_before,
                'obs_max_so_far_f': max(so_far) if so_far else None,
                'obs_sky_cover': sky_cover(obs, times, cutoff)})
    return out


def sky_cover(obs, times, when, window_hours=3):
    """Most-covered of the two lowest reported layers in the latest report with a sky code."""
    i = bisect.bisect_right(times, when)
    while i > 0:
        i -= 1
        t, row = obs[i]
        if when - t > dt.timedelta(hours=window_hours):
            return None
        layers = [CLOUD[row[c]] for c in ('skyc1', 'skyc2') if row.get(c) in CLOUD]
        if layers:
            return max(layers)
    return None


# ---------------------------------------------------------------- assemble

def ghcn_highs():
    with (DATA / 'historical_tmax.csv').open(newline='') as f:
        return {r['date']: int(r['rounded_tmax_f']) for r in csv.DictReader(f) if r['quality_flag'] == ''}


def settlement_highs():
    """Kalshi settlement high where it exists; it takes precedence over GHCN in the window."""
    with (DATA / 'matched.csv').open(newline='', encoding='utf-8') as f:
        return {r['target_date']: float(r['kalshi_settlement_high_f']) for r in csv.DictReader(f)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=8, help='parallel ECMWF downloads')
    parser.add_argument('--start', type=dt.date.fromisoformat, default=HISTORY_START)
    parser.add_argument('--end', type=dt.date.fromisoformat, default=config.WINDOW_END)
    args = parser.parse_args()
    days = days_between(args.start, args.end)

    print(f'MOS ({", ".join(MOS_MODELS)}) and ASOS ({", ".join(ASOS_STATIONS)}) from IEM...')
    mos = {m: mos_runs(m, args.start - dt.timedelta(days=3), args.end) for m in MOS_MODELS}
    asos = {}
    for station in ASOS_STATIONS:
        obs = asos_obs(station, args.start - dt.timedelta(days=3), args.end)
        asos[station] = (obs, [t for t, _ in obs])

    print(f'ECMWF for {len(days)} days...')
    with ThreadPoolExecutor(args.workers) as pool:
        ecmwf = {}
        for i, row in enumerate(pool.map(ecmwf_day, days)):
            ecmwf[row['target_date']] = row
            if i % 50 == 0:
                print(f'  {row["target_date"]} {row.get("ecmwf_max_f", row.get("ecmwf_error"))}', flush=True)

    ghcn, settled = ghcn_highs(), settlement_highs()
    rows = []
    for day in days:
        key = day.isoformat()
        high = settled.get(key, ghcn.get(key))
        if high is None:
            continue
        row = {'target_date': key, 'cutoff_utc': cutoff_for(day).isoformat(), 'split': config.split_of(day) or 'history',
               'high_f': high, 'high_source': 'kalshi' if key in settled else 'ghcn',
               'ghcn_high_d2_f': ghcn.get((day - dt.timedelta(days=2)).isoformat())}
        row.update({k: v for k, v in ecmwf[key].items() if k != 'target_date'})
        for model, runs in mos.items():
            row.update(mos_features(model, runs, day))
        for station, (obs, times) in asos.items():
            row.update(asos_features(station, obs, times, day))
        rows.append(row)
    columns = list(dict.fromkeys(k for r in rows for k in r))
    write_csv('features/daily_features.csv', [{c: r.get(c, '') for c in columns} for r in rows])
    coverage = {c: sum(r.get(c) not in (None, '') for r in rows) for c in columns}
    (DATA / 'features' / 'coverage.json').write_text(json.dumps({'days': len(rows), 'non_missing': coverage}, indent=2))
    print(f'{len(rows)} days -> data/features/daily_features.csv')


if __name__ == '__main__':
    main()
