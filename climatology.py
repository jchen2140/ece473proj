"""Seasonal empirical daily-high baseline for NYC Central Park; standard library only."""
import argparse
import calendar
import csv
import datetime as dt
import hashlib
import json
import math
import statistics
from collections import Counter

from collect_nyc import DATA, hits, write_csv


def seasonal_distance(a, b):
    """Circular month/day distance on a 366-day reference calendar."""
    x = dt.date(2000, a.month, a.day).timetuple().tm_yday
    y = dt.date(2000, b.month, b.day).timetuple().tm_yday
    delta = abs(x-y)
    return min(delta, 366-delta)


def quantile(values, probability):
    """Linear interpolation between ordered samples, including endpoints."""
    if not values or not 0 <= probability <= 1:
        raise ValueError('Quantile requires samples and a probability in [0, 1]')
    ordered = sorted(values)
    index = (len(ordered)-1)*probability
    lo, hi = math.floor(index), math.ceil(index)
    return ordered[lo] + (ordered[hi]-ordered[lo])*(index-lo)


class SeasonalClimatology:
    def __init__(self, history, start_year=1991, end_year=2020, window_days=15):
        if start_year > end_year or not 0 <= window_days <= 182:
            raise ValueError('Invalid training years or calendar window')
        self.start_year, self.end_year, self.window_days = start_year, end_year, window_days
        # Filtering happens before fitting: adding post-training outcomes cannot change predictions.
        self.records = [r for r in history if start_year <= int(r['date'][:4]) <= end_year]
        if not self.records:
            raise ValueError('No observations in the requested training period')
        if len({r['date'] for r in self.records}) != len(self.records):
            raise ValueError('Duplicate training dates')
        for r in self.records:
            dt.date.fromisoformat(r['date'])
            if r.get('ghcn_station_id') != 'USW00094728' or r.get('quality_flag'):
                raise ValueError('Expected quality-screened Central Park records')
            value = float(r['tmax_f'])
            if not math.isfinite(value) or int(r['rounded_tmax_f']) != math.floor(value+0.5):
                raise ValueError('Invalid temperature or inconsistent rounding')
        self.dates = {r['date']: dt.date.fromisoformat(r['date']) for r in self.records}
        self.calendar_days = {key: dt.date(2000, value.month, value.day).timetuple().tm_yday
                              for key, value in self.dates.items()}

    def sample(self, target):
        if target.year <= self.end_year:
            raise ValueError('Prediction year must follow the entire fixed training period')
        day = dt.date(2000, target.month, target.day).timetuple().tm_yday
        sample = [r for r in self.records
                  if min(abs(self.calendar_days[r['date']]-day), 366-abs(self.calendar_days[r['date']]-day)) <= self.window_days]
        if not sample:
            raise ValueError('No observations in the seasonal window')
        return sample

    def distribution(self, target):
        counts = Counter(int(r['rounded_tmax_f']) for r in self.sample(target))
        total = sum(counts.values())
        return [{'temperature_f': value, 'count': counts[value], 'probability': counts[value]/total}
                for value in sorted(counts)]

    def cdf(self, target, temperature_f):
        return sum(r['probability'] for r in self.distribution(target) if r['temperature_f'] <= temperature_f)

    def probability(self, target, contract):
        sample = self.sample(target)
        return sum(hits(contract, int(r['rounded_tmax_f'])) for r in sample)/len(sample)

    def summary(self, target):
        sample = self.sample(target)
        values = [float(r['tmax_f']) for r in sample]
        return {'sample_count': len(values), 'sample_year_count': len({r['date'][:4] for r in sample}),
                'mean_high_f': statistics.mean(values), 'stddev_high_f': statistics.pstdev(values),
                'p10_high_f': quantile(values, 0.1), 'median_high_f': quantile(values, 0.5),
                'p90_high_f': quantile(values, 0.9)}


def load_history():
    with (DATA/'historical_tmax.csv').open(newline='') as f:
        return list(csv.DictReader(f))


def build_outputs(model, history):
    training_name = f'climatology/training_tmax_{model.start_year}_{model.end_year}.csv'
    write_csv(training_name, model.records)
    post = [r for r in history if int(r['date'][:4]) > model.end_year]
    write_csv('climatology/post_training_tmax.csv', post)
    year_rows = []
    missing_dates = []
    accepted = {r['date'] for r in model.records}
    for year in range(model.start_year, model.end_year+1):
        start = dt.date(year, 1, 1)
        expected = 366 if calendar.isleap(year) else 365
        dates = [(start+dt.timedelta(days=i)).isoformat() for i in range(expected)]
        missing = [d for d in dates if d not in accepted]
        missing_dates.extend(missing)
        values = [float(r['tmax_f']) for r in model.records if r['date'][:4] == str(year)]
        year_rows.append({'year': year, 'expected_days': expected, 'accepted_days': len(values),
                          'missing_or_rejected_days': len(missing), 'coverage_fraction': len(values)/expected,
                          'mean_daily_high_f': statistics.mean(values) if values else None})
    write_csv('climatology/yearly_coverage.csv', year_rows)
    seasonal = []
    # Use leap year 2024 solely as a month/day template, not as input observations.
    for i in range(366):
        target = dt.date(2024, 1, 1)+dt.timedelta(days=i)
        seasonal.append({'month_day': target.strftime('%m-%d'), **model.summary(target)})
    write_csv('climatology/seasonal_summary.csv', seasonal)
    monthly = []
    for month in range(1, 13):
        values = [float(r['tmax_f']) for r in model.records if int(r['date'][5:7]) == month]
        monthly.append({'month': month, 'sample_count': len(values), 'mean_high_f': statistics.mean(values),
                        'stddev_high_f': statistics.pstdev(values), 'p10_high_f': quantile(values, 0.1),
                        'median_high_f': quantile(values, 0.5), 'p90_high_f': quantile(values, 0.9)})
    write_csv('climatology/monthly_summary.csv', monthly)
    source = json.loads((DATA/'raw/ghcn/USW00094728.dly.source.json').read_text())
    manifest = {'station_id': 'USW00094728', 'training_years': [model.start_year, model.end_year],
                'calendar_window_days_each_side': model.window_days, 'training_records': len(model.records),
                'expected_training_days': sum(r['expected_days'] for r in year_rows),
                'missing_or_rejected_dates': missing_dates, 'post_training_records': len(post),
                'training_csv_sha256': hashlib.sha256((DATA/training_name).read_bytes()).hexdigest(),
                'source': source,
                'method': f'Empirical distribution of rounded daily TMAX in a +/-{model.window_days}-day seasonal window, {model.start_year}-{model.end_year} only.',
                'leap_day_policy': 'Month/day distance on circular 366-day reference year; non-leap years contribute no February 29 record.',
                'limitations': ['Current revised GHCN archive, not an as-issued historical snapshot.',
                               'No trend, bias or spread adjustment fitted.',
                               'Historical station daily maxima, not the exact original CLI settlement releases.']}
    (DATA/'climatology/manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--date', help='Forecast target date after 2020 (YYYY-MM-DD)')
    parser.add_argument('--lower', type=int, help='Inclusive whole-degree lower bound; omit for lower tail')
    parser.add_argument('--upper', type=int, help='Inclusive whole-degree upper bound; omit for upper tail')
    args = parser.parse_args()
    if not args.date and (args.lower is not None or args.upper is not None):
        parser.error('Bounds require --date')
    history = load_history()
    model = SeasonalClimatology(history)
    if args.date:
        target = dt.date.fromisoformat(args.date)
        result = {'target_date': args.date, 'training_years': [1991, 2020], **model.summary(target),
                  'distribution': model.distribution(target)}
        if args.lower is not None or args.upper is not None:
            if args.lower is not None and args.upper is not None and args.lower > args.upper:
                parser.error('--lower cannot exceed --upper')
            result['inclusive_range_f'] = [args.lower, args.upper]
            result['range_probability'] = sum(r['probability'] for r in result['distribution']
                if (args.lower is None or r['temperature_f'] >= args.lower)
                and (args.upper is None or r['temperature_f'] <= args.upper))
        print(json.dumps(result, indent=2))
    else:
        print(json.dumps(build_outputs(model, history), indent=2))


if __name__ == '__main__':
    main()
