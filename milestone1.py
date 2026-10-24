"""Milestone 1 weeks 1–2: station history, climatology, and forecast evaluation."""
import calendar
import csv
import datetime as dt
import json
import math
from collections import defaultdict

from collect_nyc import DATA, fetch, hits, write_csv

GHCN = 'https://www.ncei.noaa.gov/pub/data/ghcn/daily'
STATION = 'USW00094728'


def cached_text(filename, url):
    path = DATA / 'raw' / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        body, headers, _ = fetch(url)
        path.write_bytes(body)
        path.with_suffix(path.suffix + '.source.json').write_text(json.dumps({
            'url': url, 'retrieved_at': dt.datetime.now(dt.timezone.utc).isoformat(),
            'last_modified': headers.get('Last-Modified') or headers.get('last-modified'),
        }, indent=2))
    return path.read_text()


def parse_tmax(text):
    """Parse GHCN .dly monthly fixed-width records; retain only unflagged TMAX."""
    rows = []
    rejected = 0
    for line in text.splitlines():
        if line[17:21] != 'TMAX':
            continue
        if line[:11] != STATION:
            raise ValueError('Wrong GHCN station')
        year, month = int(line[11:15]), int(line[15:17])
        for day in range(1, calendar.monthrange(year, month)[1] + 1):
            field = line[21 + (day-1)*8:21 + day*8]
            raw = int(field[:5])
            if raw == -9999 or field[6] != ' ':
                rejected += 1
                continue
            temperature = raw/10*9/5+32
            rows.append({'date': dt.date(year, month, day).isoformat(), 'ghcn_station_id': STATION,
                         'tmax_tenths_c': raw, 'tmax_f': round(temperature, 4),
                         'rounded_tmax_f': math.floor(temperature+0.5),
                         'measurement_flag': field[5].strip(), 'quality_flag': '',
                         'source_flag': field[7].strip()})
    if len({r['date'] for r in rows}) != len(rows):
        raise ValueError('Duplicate station dates')
    return sorted(rows, key=lambda r: r['date']), rejected


def seasonal_distance(a, b):
    # Reference leap year keeps month/day alignment stable across calendar years.
    x = dt.date(2000, a.month, a.day).timetuple().tm_yday
    y = dt.date(2000, b.month, b.day).timetuple().tm_yday
    delta = abs(x-y)
    return min(delta, 366-delta)


def training_sample(history, target):
    """Fixed 1991–2020 baseline, +/-15 calendar days, excluding target/future data."""
    return [r for r in history
            if 1991 <= int(r['date'][:4]) <= 2020
            and dt.date.fromisoformat(r['date']) < target
            and seasonal_distance(dt.date.fromisoformat(r['date']), target) <= 15]


def read_csv(name):
    with (DATA/name).open(newline='') as f:
        return list(csv.DictReader(f))


def strike(row):
    return {'strike_type': row['strike_type'],
            'floor_strike': float(row['lower_bound']) if row['lower_bound'] else None,
            'cap_strike': float(row['upper_bound']) if row['upper_bound'] else None}


def score(probabilities, outcomes):
    if not probabilities or len(probabilities) != len(outcomes):
        raise ValueError('Evaluation requires aligned nonempty rows')
    if any(not 0 <= p <= 1 for p in probabilities):
        raise ValueError('Invalid probability')
    # Clipping affects log loss only; Brier uses original empirical probabilities.
    eps = 1e-6
    clipped = [min(1-eps, max(eps, p)) for p in probabilities]
    return {'binary_contract_count': len(outcomes),
            'brier_mean_per_binary_contract': sum((p-y)**2 for p,y in zip(probabilities,outcomes))/len(outcomes),
            'binary_log_loss_mean': -sum(y*math.log(p)+(1-y)*math.log(1-p) for p,y in zip(clipped,outcomes))/len(outcomes),
            'log_loss_clip_epsilon': eps}


def main():
    stations = cached_text('ghcn/ghcnd-stations.txt', GHCN+'/ghcnd-stations.txt')
    station_line = next(line for line in stations.splitlines() if line.startswith(STATION))
    if not any(name in station_line[41:71].upper() for name in ('CENTRAL PARK', 'CNTRL PARK')):
        raise ValueError('Station metadata does not identify Central Park')
    station = {'station_id': STATION, 'latitude': float(station_line[12:20]),
               'longitude': float(station_line[21:30]), 'name': station_line[41:71].strip(),
               'original_metadata_record': station_line}
    (DATA/'ghcn_station.json').write_text(json.dumps(station, indent=2))
    history, rejected = parse_tmax(cached_text(f'ghcn/{STATION}.dly', f'{GHCN}/all/{STATION}.dly'))
    history = [r for r in history if '1991-01-01' <= r['date'] <= '2026-12-31']
    write_csv('historical_tmax.csv', history)
    matched = read_csv('matched.csv')
    groups = defaultdict(list)
    for row in matched:
        groups[row['target_date']].append(row)
    enriched, baseline_info, observation_checks = [], [], []
    history_by_date = {r['date']: r for r in history}
    for date, rows in sorted(groups.items()):
        sample = training_sample(history, dt.date.fromisoformat(date))
        years = {r['date'][:4] for r in sample}
        if len(sample) < 800 or len(years) != 30:
            raise ValueError(f'Insufficient historical climatology for {date}')
        assert max(r['date'] for r in sample) < date
        probabilities = []
        for row in rows:
            p = sum(hits(strike(row), r['rounded_tmax_f']) for r in sample)/len(sample)
            probabilities.append(p)
            enriched.append({**row, 'climatology_probability': p, 'climatology_sample_count': len(sample),
                             'climatology_training_start_year': 1991, 'climatology_training_end_year': 2020,
                             'climatology_calendar_window_days': 15})
        if not math.isclose(sum(probabilities), 1, abs_tol=1e-12):
            raise ValueError('Climatology probabilities do not partition into bins')
        baseline_info.append({'target_date': date, 'training_sample_count': len(sample),
                              'training_year_count': len(years), 'latest_training_date': max(r['date'] for r in sample)})
        observed = history_by_date.get(date)
        high = float(rows[0]['kalshi_settlement_high_f'])
        observation_checks.append({'target_date': date, 'kalshi_settlement_high_f': high,
                                   'ghcn_tmax_f': observed['tmax_f'] if observed else None,
                                   'ghcn_rounded_tmax_f': observed['rounded_tmax_f'] if observed else None,
                                   'rounded_values_agree': observed['rounded_tmax_f'] == high if observed else None,
                                   'note': 'GHCN is an independent archive cross-check; original settlement CLI release not retrieved.'})
    write_csv('matched_with_climatology.csv', enriched)
    write_csv('climatology_training_summary.csv', baseline_info)
    write_csv('observation_crosscheck.csv', observation_checks)
    # Score all three forecasts on exactly the same rows.
    shared = [r for r in enriched if r['market_midpoint'] != '']
    outcomes = [int(r['outcome']) for r in shared]
    metrics, calibration = {}, []
    for model, column in [('ensemble', 'ensemble_probability'), ('climatology', 'climatology_probability'), ('market', 'market_midpoint')]:
        ps = [float(r[column]) for r in shared]
        metrics[model] = score(ps, outcomes)
        metrics[model]['event_count'] = len({r['target_date'] for r in shared})
        for i in range(10):
            indexes = [j for j,p in enumerate(ps) if min(9, int(p*10)) == i]
            calibration.append({'model': model, 'bin_lower': i/10, 'bin_upper': (i+1)/10,
                                'count': len(indexes),
                                'mean_predicted_probability': sum(ps[j] for j in indexes)/len(indexes) if indexes else None,
                                'observed_yes_frequency': sum(outcomes[j] for j in indexes)/len(indexes) if indexes else None})
    write_csv('calibration.csv', calibration)
    report = {'evaluation_dates': sorted(groups), 'shared_contracts': len(shared),
              'excluded_missing_market_quotes': len(enriched)-len(shared),
              'climatology': {'station': station, 'training_years': [1991, 2020], 'calendar_window': '+/-15 days',
                              'accepted_history_rows_1991_onward': len(history), 'rejected_missing_or_quality_flagged_all_years': rejected},
              'metrics': metrics,
              'cautions': ['Seven events is a smoke test, not a reliable ranking of models.',
                           'Six contracts in each event are dependent; confidence intervals must resample events/days.',
                           'Retrospectively revised GHCN records are used for historical climatology; this is not a vintage-locked archive.',
                           'Probabilities are uncalibrated; no bias/spread fit or market-making simulation is included.',
                           'GHCN is not a replacement for the exact NWS report used in settlement.']}
    (DATA/'evaluation.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print('Observation cross-check:', observation_checks)


if __name__ == '__main__':
    main()
