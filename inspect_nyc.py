"""Check the saved CSVs without downloading anything or importing GRIB libraries."""
import csv
import datetime as dt
import json
from collections import defaultdict
from pathlib import Path

import config

DATA = Path(__file__).resolve().parent / 'data'


def load(name):
    with (DATA / name).open(newline='') as f:
        return list(csv.DictReader(f))


def main():
    matched = load('matched.csv')
    points = load('ensemble_temperatures.csv')
    maxima = load('ensemble_daily_maxima.csv')
    groups = defaultdict(list)
    for row in matched:
        groups[row['target_date']].append(row)
        cutoff = dt.datetime.fromisoformat(row['prediction_time_utc'])
        assert dt.datetime.fromisoformat(row['model_run_utc']) < cutoff
        if row['market_candle_end_utc']:
            quote = dt.datetime.fromisoformat(row['market_candle_end_utc'])
            assert dt.timedelta() <= cutoff - quote <= dt.timedelta(hours=2)
    summaries = []
    for day, rows in sorted(groups.items()):
        assert len(rows) == 6
        assert sum(int(r['outcome']) for r in rows) == 1
        assert sum(int(r['members_in_bin']) for r in rows) == 31
        assert abs(sum(float(r['ensemble_probability']) for r in rows)-1) < 1e-12
        assert len({r['kalshi_settlement_high_f'] for r in rows}) == 1
        member_rows = [r for r in maxima if r['target_date'] == day]
        sample_rows = [r for r in points if r['target_date'] == day]
        assert len(member_rows) == 31
        assert len(sample_rows) == 310
        assert len({r['member'] for r in member_rows}) == 31
        assert len({(r['member'],r['forecast_hour']) for r in sample_rows}) == 310
        cutoff = dt.datetime.fromisoformat(rows[0]['prediction_time_utc'])
        assert all(dt.datetime.fromisoformat(r['archive_last_modified']) <= cutoff for r in sample_rows)
        highs = [float(r['daily_high_approx_f']) for r in member_rows]
        summaries.append({'target_date': day, 'kalshi_settlement_high_f': rows[0]['kalshi_settlement_high_f'],
                          'ensemble_mean_high_approx_f': round(sum(highs)/31, 2),
                          'ensemble_min_high_approx_f': round(min(highs), 2),
                          'ensemble_max_high_approx_f': round(max(highs), 2),
                          'winning_bin': next(r['bin_label'] for r in rows if r['outcome']=='1'),
                          'market_midpoints_available': sum(bool(r['market_midpoint']) for r in rows)})
    with (DATA/'daily_summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(summaries)
    report = json.loads((DATA/'validation.json').read_text())
    assert report['contracts'] == len(matched)
    assert report['days'] == len(groups)
    assert report['temperature_points'] == len(points)
    # Collected and skipped days together account for every requested day, with no overlap.
    skipped = load('skipped_days.csv') if (DATA/'skipped_days.csv').exists() else []
    skipped_days = {r['target_date'] for r in skipped}
    assert not skipped_days & set(groups)
    first = dt.date.fromisoformat(report['requested_start'])
    requested = {(first + dt.timedelta(days=i)).isoformat() for i in range(report['requested_days'])}
    assert set(groups) | skipped_days == requested
    assert all(len({r['market_ticker'] for r in rows}) == 6 for rows in groups.values())
    in_window = [d for d in groups if config.WINDOW_START <= dt.date.fromisoformat(d) <= config.WINDOW_END]
    splits = defaultdict(int)
    for d in groups:
        splits[config.split_of(d)] += 1
    print(f'PASS: {len(groups)} days; {len(matched)} contracts; {len(maxima)} member maxima; {len(points)} forecast points.')
    print(f'{len(skipped)} skipped days; {len(in_window)} collected days inside the config window; '
          f'train {splits["train"]}, test {splits["test"]}, outside {splits[None]}.')
    for row in skipped:
        print('SKIPPED', row['target_date'], row['reason'], row['detail'])
    for row in summaries if len(summaries) <= 6 else summaries[:3] + ['...'] + summaries[-3:]:
        print(row)


if __name__ == '__main__':
    main()
