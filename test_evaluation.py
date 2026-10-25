import datetime as dt
import random
import unittest
from unittest import mock

import config
import collect_nyc
import evaluate
from collect_nyc import SkipDay, hits
from persistence import Persistence

BINS = [{'strike_type': 'less', 'cap_strike': 77},
        {'strike_type': 'between', 'floor_strike': 77, 'cap_strike': 78},
        {'strike_type': 'between', 'floor_strike': 79, 'cap_strike': 80},
        {'strike_type': 'between', 'floor_strike': 81, 'cap_strike': 82},
        {'strike_type': 'between', 'floor_strike': 83, 'cap_strike': 84},
        {'strike_type': 'greater', 'floor_strike': 84}]


def record(date, high):
    return {'date': date, 'ghcn_station_id': 'USW00094728', 'tmax_f': high,
            'rounded_tmax_f': int(high), 'quality_flag': ''}


class ConfigChecks(unittest.TestCase):
    def test_split_boundaries(self):
        self.assertEqual(config.split_of('2025-08-14'), 'train')
        self.assertEqual(config.split_of('2026-04-30'), 'train')
        self.assertEqual(config.split_of('2026-05-01'), 'test')
        self.assertEqual(config.split_of('2026-08-13'), 'test')
        self.assertIsNone(config.split_of('2026-08-14'))
        self.assertIsNone(config.split_of('2025-08-13'))
        self.assertEqual(config.window_days(), 365)


class BootstrapChecks(unittest.TestCase):
    def rows(self):
        rng = random.Random(2)
        out = []
        for d in range(30):
            win = rng.randrange(6)
            for k in range(6):
                p = rng.random() / 3
                out.append({'target_date': f'2026-06-{d+1:02d}', 'outcome': int(k == win),
                            'ensemble_probability': p, 'climatology_probability': 1 / 6})
        return out

    def test_point_estimate_is_pooled_mean_and_interval_brackets_it(self):
        rows = self.rows()
        result = evaluate.bootstrap(rows, ['raw_ensemble', 'climatology'], [('raw_ensemble', 'climatology')],
                                    replicates=500)
        brier = [(float(r['ensemble_probability']) - r['outcome']) ** 2 for r in rows]
        cell = result['models']['raw_ensemble']['brier']
        self.assertAlmostEqual(cell['mean'], sum(brier) / len(brier))
        self.assertLess(cell['ci_low'], cell['mean'])
        self.assertGreater(cell['ci_high'], cell['mean'])
        self.assertEqual((result['days'], result['contracts']), (30, 180))
        diff = result['pairs']['raw_ensemble-climatology']['brier']
        self.assertAlmostEqual(diff['mean'], cell['mean'] - result['models']['climatology']['brier']['mean'])

    def test_identical_models_have_zero_difference(self):
        rows = self.rows()
        for r in rows:
            r['climatology_probability'] = r['ensemble_probability']
        diff = evaluate.bootstrap(rows, ['raw_ensemble', 'climatology'], [('raw_ensemble', 'climatology')],
                                  replicates=200)['pairs']['raw_ensemble-climatology']['brier']
        self.assertEqual((diff['mean'], diff['ci_low'], diff['ci_high']), (0, 0, 0))


class PersistenceChecks(unittest.TestCase):
    def test_lag_two_anchor_and_no_future_changes(self):
        history = [record('2020-07-18', 80), record('2020-07-20', 83),
                   record('2026-07-18', 90), record('2026-07-20', 50)]
        model = Persistence(history)
        target = dt.date(2026, 7, 20)
        self.assertEqual(model.anchor(target), 90)  # July 18 is the latest complete day at noon July 19
        self.assertEqual(dict(model.change_distribution(target)), {3: 1})  # 2026 change excluded
        self.assertEqual(model.probability(target, {'strike_type': 'greater', 'floor_strike': 92}), 1)
        self.assertIsNone(model.probability(dt.date(2026, 7, 25), BINS[0]))

    def test_probabilities_partition(self):
        history = [record(f'2020-07-{d:02d}', 70 + d % 9) for d in range(1, 31)] + [record('2026-07-10', 80)]
        model = Persistence(history)
        self.assertAlmostEqual(sum(model.probability(dt.date(2026, 7, 12), b) for b in BINS), 1)


class CollectorSkipChecks(unittest.TestCase):
    def market(self, ticker, rules, event='KXHIGHNY-26JUL20'):
        return {'ticker': ticker, 'event_ticker': event, 'result': 'no', 'rules_primary': rules}

    def run_get_markets(self, markets):
        def fake(name, url):
            return {'markets': markets if name.endswith('historical_markets_0.json') else [], 'cursor': ''}
        with mock.patch.object(collect_nyc, 'cached_json', side_effect=fake):
            return collect_nyc.get_markets(dt.date(2026, 7, 20))

    def test_wrong_bin_count_is_a_skip(self):
        nws = 'Central Park ... National Weather Service'
        with self.assertRaises(SkipDay) as ctx:
            self.run_get_markets([self.market(f'T{i}', nws) for i in range(5)])
        self.assertEqual(ctx.exception.reason, 'wrong_bin_count')

    def test_non_nws_rules_and_wrong_event_are_skips(self):
        with self.assertRaises(SkipDay) as ctx:
            self.run_get_markets([self.market(f'T{i}', 'according to The Weather Company') for i in range(6)])
        self.assertEqual(ctx.exception.reason, 'non_nws_rules')
        with self.assertRaises(SkipDay) as ctx:
            self.run_get_markets([self.market('T0', 'x', event='KXHIGHNY-25OCT15')])
        self.assertEqual(ctx.exception.reason, 'unexpected_event')


class SettlementFallbackChecks(unittest.TestCase):
    """collect_day with fake markets and a flat 80°F forecast; no network access."""

    def pairs(self, expiration):
        markets = []
        for b in BINS:
            label = f"{b['strike_type']}{b.get('floor_strike')}{b.get('cap_strike')}"
            markets.append(({**b, 'ticker': label, 'event_ticker': 'KXHIGHNY-25NOV07', 'yes_sub_title': label,
                             'result': 'yes' if hits(b, 79) else 'no', 'expiration_value': expiration,
                             'rules_primary': 'Central Park National Weather Service'}, True))
        return markets

    def collect(self, expiration, ghcn):
        def point(task):
            day, member, lead, *_ = task
            return {'target_date': day.isoformat(), 'model_run_utc': '2025-11-06T06:00:00+00:00', 'member': member,
                    'forecast_hour': lead, 'temperature_f': 80.0}
        with mock.patch.object(collect_nyc, 'get_markets', return_value=self.pairs(expiration)), \
                mock.patch.object(collect_nyc, 'candle_at', return_value=(None, None, None, None)), \
                mock.patch.object(collect_nyc, 'gefs_point', side_effect=point), \
                mock.patch.object(collect_nyc, 'ghcn_rounded_high', return_value=ghcn):
            return collect_nyc.collect_day(dt.date(2025, 11, 7), 40.78, -73.97, 2)

    def test_blank_settlement_uses_consistent_ghcn_high(self):
        result = self.collect('', 79)
        self.assertEqual({r['settlement_high_source'] for r in result['matched']}, {'ghcn_fallback'})
        self.assertEqual({r['kalshi_settlement_high_f'] for r in result['matched']}, {'79.00'})
        self.assertEqual(sum(r['members_in_bin'] for r in result['matched']), 31)

    def test_blank_settlement_with_inconsistent_or_missing_ghcn_is_skipped(self):
        for ghcn, reason in [(85, 'ghcn_fallback_inconsistent'), (None, 'missing_settlement_high')]:
            with self.assertRaises(SkipDay) as ctx:
                self.collect('', ghcn)
            self.assertEqual(ctx.exception.reason, reason)

    def test_partly_blank_settlement_uses_the_reported_value(self):
        pairs = self.pairs('79.00')
        pairs[0][0]['expiration_value'] = ''
        with mock.patch.object(collect_nyc, 'get_markets', return_value=pairs), \
                mock.patch.object(collect_nyc, 'candle_at', return_value=(None, None, None, None)), \
                mock.patch.object(collect_nyc, 'gefs_point', side_effect=lambda task: {
                    'target_date': '2025-11-07', 'model_run_utc': 'x', 'member': task[1],
                    'forecast_hour': task[2], 'temperature_f': 80.0}):
            result = collect_nyc.collect_day(dt.date(2025, 11, 7), 40.78, -73.97, 2)
        self.assertEqual({r['kalshi_settlement_high_f'] for r in result['matched']}, {'79.00'})
        self.assertEqual({r['settlement_high_source'] for r in result['matched']}, {'kalshi'})

    def test_reported_settlement_is_kept(self):
        result = self.collect('79.00', 60)
        self.assertEqual({r['settlement_high_source'] for r in result['matched']}, {'kalshi'})


if __name__ == '__main__':
    unittest.main()
