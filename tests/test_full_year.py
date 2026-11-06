"""Checks for market/full_year.py and market/risk_frontier.py: walk-forward leakage, true tail bounds,
and the capital identities the risk analysis relies on."""
import math
import unittest

import numpy as np
import pandas as pd

from market import full_year, makers, risk_frontier
from tests.test_makers import synthetic_day


def grid_rows(days_pnl):
    """days_pnl: {day: {b: pnl}} for the forecast maker at one fee."""
    return pd.DataFrame([{'city': 'X', 'day': d, 'maker': 'forecast', 'b': b, 'fee': 0.05, 'margin_f': 0.0, 'pnl': p}
                         for d, by_b in days_pnl.items() for b, p in by_b.items()])


class WalkForwardTest(unittest.TestCase):
    def test_fit_cutoff_is_two_days_before_the_month(self):
        # The first quote of October is noon on Sep 30, when Sep 30's high is not final.
        self.assertEqual(full_year.fit_cutoff('2025-10'), '2025-09-29')
        self.assertEqual(full_year.fit_cutoff('2026-03'), '2026-02-27')

    def test_test_months_start_in_october(self):
        days = [{'day': d} for d in ('2025-08-20', '2025-09-30', '2025-10-01', '2026-08-13')]
        self.assertEqual(full_year.test_months(days), ['2025-10', '2026-08'])

    def test_choice_ignores_days_after_the_cutoff(self):
        # Past days favour b=30; the test month itself would favour b=1000 by a mile.
        grid = grid_rows({'2025-09-01': {30: 10, 1000: 1}, '2025-09-29': {30: 10, 1000: 1},
                          '2025-09-30': {30: 0, 1000: 9999}, '2025-10-05': {30: 0, 1000: 9999}})
        b, fee, n = full_year.choose(grid, 'forecast', '2025-09-29')
        self.assertEqual((b, fee, n), (30, 0.05, 2))

    def test_ties_go_to_the_smaller_b(self):
        grid = grid_rows({'2025-09-01': {30: 5, 100: 5, 300: 5}})
        self.assertEqual(full_year.choose(grid, 'forecast', '2025-09-29')[0], 30)

    def test_margin_uses_only_final_days(self):
        audit = [{'day': f'2025-09-{d:02d}', 'eligible': 1, 'sensor_minus_official_f': 0.4} for d in range(1, 29)]
        audit.append({'day': '2025-09-30', 'eligible': 1, 'sensor_minus_official_f': 5.0})  # Not final by Oct's first quote.
        fit = full_year.fit_margin(audit, full_year.fit_cutoff('2025-10'))
        self.assertEqual(fit['margin_f'], 0.5)
        self.assertEqual(fit['latest_fit_date'], '2025-09-28')


class TrueBoundsTest(unittest.TestCase):
    def test_open_top_bin_is_never_written_off(self):
        """With the true +inf upper edge, a running maximum far above the top bin's floor leaves it alive."""
        day = synthetic_day(np.random.default_rng(5), 5)
        day['running_max'] = np.full(len(day['minute_ts']), 99.0)
        weights, _ = makers.tilt(day, day['minute_ts'][-1] + makers.FEED_DELAY)
        self.assertEqual(weights[5], 1.0)
        self.assertTrue(np.all(weights[:5] == makers.DEAD_WEIGHT))

    def test_finite_placeholder_would_write_off_the_winner(self):
        """The bug the true bounds avoid: a placeholder top edge (floor + 3F) kills the winning top bin."""
        day = synthetic_day(np.random.default_rng(5), 5)
        day['highs'] = day['highs'].copy()
        day['highs'][5] = 89.5 + 3.0
        day['running_max'] = np.full(len(day['minute_ts']), 99.0)
        weights, _ = makers.tilt(day, day['minute_ts'][-1] + makers.FEED_DELAY)
        self.assertEqual(weights[5], makers.DEAD_WEIGHT)


class SpikeFilterTest(unittest.TestCase):
    def day_with_readings(self, readings):
        day = synthetic_day(np.random.default_rng(7), 5)
        day['minute_ts'] = day['start'] + 60.0 * np.arange(len(readings))
        day['running_max'] = np.maximum.accumulate(np.asarray(readings, dtype=float))
        series = pd.Series(readings, index=pd.to_datetime(day['minute_ts'], unit='s'), dtype=float)
        day['running_max_filtered'] = np.fmax.accumulate(series.rolling(full_year.SPIKE_WINDOW).median().to_numpy())
        return day

    def test_a_lone_spike_does_not_write_off_bins(self):
        day = self.day_with_readings([80.0] * 30 + [95.0] + [80.0] * 30)
        end = day['minute_ts'][-1] + makers.FEED_DELAY
        self.assertTrue(np.all(full_year.filtered_tilt(day, end)[0][2:] == 1.0))
        self.assertEqual(makers.tilt(day, end)[0][4], makers.DEAD_WEIGHT)  # The raw maximum would.

    def test_a_sustained_rise_still_truncates(self):
        day = self.day_with_readings([80.0] * 30 + [88.0] * 10)
        weights, m = full_year.filtered_tilt(day, day['minute_ts'][-1] + makers.FEED_DELAY)
        self.assertEqual(m, 88.0)
        # Edges 81.5-85.5 are cleared by more than 1F; 87.5 is only touched.
        np.testing.assert_allclose(weights, [makers.DEAD_WEIGHT] * 3 + [makers.TOUCHED_WEIGHT, 1, 1])

    def test_filter_uses_no_future_readings(self):
        day = self.day_with_readings([80.0] * 30 + [88.0] * 10)
        weights, m = full_year.filtered_tilt(day, day['minute_ts'][29] + makers.FEED_DELAY)
        self.assertEqual(m, 80.0)


class TailTest(unittest.TestCase):
    def test_worst_share_mean_uses_the_ceiling_of_five_percent(self):
        x = list(range(100, 121))  # 21 days -> ceil(1.05) = 2 worst days.
        self.assertEqual(full_year.worst_share_mean(x), 100.5)


class CapitalTest(unittest.TestCase):
    def test_static_loss_bound_holds_for_every_maker_prior(self):
        rng = np.random.default_rng(11)
        for trial in range(100):
            winner = int(rng.integers(6))
            day = synthetic_day(rng, winner)
            day['priors']['uniform'] = np.full(6, 1 / 6)
            for maker in ('generic', 'forecast'):
                for b in (10.0, 300.0):
                    r = full_year.replay_one(day, maker, b, 0.0, 2.0, 0.0)
                    self.assertLessEqual(-r['trading_pnl'], r['opening_bound'] + 1e-6, f'{trial} {maker} {b}')
                    self.assertLessEqual(r['opening_bound'], r['max_opening_bound'] + 1e-9)

    def test_expected_capital_is_b_times_log_loss(self):
        rng = np.random.default_rng(3)
        priors = rng.dirichlet(np.ones(6), size=50)
        winners = rng.integers(6, size=50)
        b = 300.0
        bounds = [b * math.log(1 / p[w]) for p, w in zip(priors, winners)]
        log_loss = -np.mean([math.log(p[w]) for p, w in zip(priors, winners)])
        self.assertAlmostEqual(np.mean(bounds), b * log_loss)

    def test_equal_capital_interpolates_on_log_capital(self):
        rows = [{'city': 'X', 'maker': 'forecast', 'b': b, 'expected_capital': b * 1.0, 'mean_pnl': p, 'prior_log_loss': 1.0}
                for b, p in ((10, 1.0), (1000, 3.0))]
        out = {r['budget']: r for r in risk_frontier.equal_capital(rows)}
        share = math.log(150 / 10) / math.log(1000 / 10)
        self.assertAlmostEqual(out[150]['mean_pnl'], 1.0 + 2.0 * share)
        self.assertAlmostEqual(out[150]['implied_b'], 150.0)


if __name__ == '__main__':
    unittest.main()
