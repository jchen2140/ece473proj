import unittest

import numpy as np

from market import backtest
from forecast import noaa_calibration as calibrate
from analysis import compare_forecasts
from forecast import probabilities as forecasts
from market import metrics as mm_metrics
from market import traders
from market.lmsr import LMSR


def replay_with_class(flow, priors, b, fee):
    """Slow reference: the same limit-order rule through the LMSR class."""
    m = LMSR(priors, b)
    fees = 0.0
    for _, outcome, is_yes, limit, size in flow:
        if limit - fee <= 0:
            continue
        shares, _ = m.fill_to_limit(outcome, 'yes' if is_yes else 'no', limit - fee, size)
        fees += shares * fee
    return m, fees


class Replay(unittest.TestCase):
    def flow(self, n=400):
        rng = np.random.default_rng(7)
        return [(float(i), int(rng.integers(6)), bool(rng.integers(2)), float(rng.uniform(0.02, 0.98)),
                 float(rng.uniform(1, 200))) for i in range(n)]

    def test_fast_replay_matches_lmsr_class(self):
        priors = [0.05, 0.1, 0.3, 0.3, 0.2, 0.05]
        for b, fee in ((100, 0.0), (1000, 0.02), (50, 0.01)):
            state = backtest.replay_day(self.flow(), priors, b, fee, float('inf'))
            m, fees = replay_with_class(self.flow(), priors, b, fee)
            self.assertAlmostEqual(state['revenue'], m.revenue, places=6)
            self.assertAlmostEqual(state['fee_income'], fees, places=6)
            self.assertTrue(np.allclose(state['prices'], m.prices()))
            self.assertTrue(np.allclose(state['exposure'], m.exposure()))

    def test_taker_never_pays_more_than_real_price(self):
        priors = [1 / 6] * 6
        state = backtest.replay_day([(0, 0, True, 0.3, 10_000)], priors, 100, 0.0, float('inf'))
        self.assertLessEqual(state['revenue'] / state['filled'], 0.3)

    def test_trade_after_stop_is_ignored(self):
        state = backtest.replay_day([(5.0, 0, True, 0.9, 10)], [0.5, 0.5], 100, 0.0, 1.0)
        self.assertEqual((state['filled'], state['offered']), (0.0, 0.0))

    def test_opposing_flow_at_bad_prices_does_not_fill(self):
        state = backtest.replay_day([(0, 0, True, 0.2, 10), (1, 0, False, 0.2, 10)], [0.5, 0.5], 100, 0.0, float('inf'))
        self.assertEqual(state['filled'], 0.0)


class Forecasts(unittest.TestCase):
    def rows(self):
        return [{'strike_type': 'less', 'lower_bound': '', 'upper_bound': '84'},
                {'strike_type': 'between', 'lower_bound': '84', 'upper_bound': '85'},
                {'strike_type': 'greater', 'lower_bound': '85', 'upper_bound': ''}]

    def test_bins_partition_the_rounded_scale(self):
        self.assertAlmostEqual(sum(forecasts.gaussian_bins(self.rows(), 84, 2)), 1)
        self.assertEqual(forecasts.bin_edges(self.rows()[0])[1], 83.5)
        self.assertEqual(forecasts.bin_edges(self.rows()[2])[0], 85.5)

    def test_floor_leaves_no_zero(self):
        p = forecasts.floor_probabilities([0.0, 0.5, 0.5])
        self.assertGreater(min(p), 0)
        self.assertAlmostEqual(sum(p), 1)

    def moments(self):
        return {f'2026-06-{d:02d}': (80.0 + d % 3, 4.0 + d % 2) for d in range(1, 31)}

    def test_fit_ignores_the_previous_day_and_target(self):
        moments = self.moments()
        highs = {k: v[0] + (int(k[-2:]) % 2) for k, v in moments.items()}
        for mean_method, spread in (('bias', 'errors'), ('regression', 'scaled')):
            before = calibrate.predict('2026-06-30', moments, highs, 30, mean_method, spread)
            changed = {**highs, '2026-06-30': 200.0, '2026-06-29': 200.0}
            self.assertEqual(calibrate.predict('2026-06-30', moments, changed, 30, mean_method, spread), before)

    def test_bias_correction_removes_a_constant_error(self):
        moments = self.moments()
        highs = {k: v[0] - 3 + 0.5 * (int(k[-2:]) % 2) for k, v in moments.items()}
        mu, _ = calibrate.predict('2026-06-30', moments, highs, 30, 'bias', 'errors')
        self.assertAlmostEqual(mu, moments['2026-06-30'][0] - 2.75, delta=0.05)

    def test_warm_up_returns_none(self):
        self.assertIsNone(calibrate.predict('2026-06-03', self.moments(), {}, 30, 'bias', 'errors'))

    def test_sd_has_a_floor(self):
        moments = self.moments()
        _, sd = calibrate.predict('2026-06-30', moments, {k: v[0] for k, v in moments.items()}, 30, 'bias', 'errors')
        self.assertEqual(sd, forecasts.MIN_SD)


class ForecastComparison(unittest.TestCase):
    def test_logistic_recovers_known_direction(self):
        rng = np.random.default_rng(3)
        z = rng.normal(size=4000)
        y = (rng.random(4000) < 1 / (1 + np.exp(-2 * z))).astype(float)
        beta = compare_forecasts.fit_logistic(np.column_stack([np.ones(4000), z]), y)
        self.assertAlmostEqual(beta[1], 2, delta=0.25)

    def test_slope_of_exact_line(self):
        x = np.array([0.0, 1.0, 2.0])
        self.assertAlmostEqual(compare_forecasts.slope(x, 3 * x + 1), 3)


class Traders(unittest.TestCase):
    def test_belief_order_needs_threshold(self):
        self.assertIsNone(traders.belief_order([0.5, 0.5], [0.49, 0.51], 0.03))

    def test_belief_order_direction(self):
        self.assertEqual(traders.belief_order([0.7, 0.2, 0.1], [0.5, 0.3, 0.2], 0.05)[:2], (0, 'yes'))
        outcome, side, limit = traders.belief_order([0.1, 0.5, 0.4], [0.4, 0.3, 0.3], 0.02)
        self.assertEqual((outcome, side), (0, 'no'))
        self.assertAlmostEqual(limit, 0.88)

    def test_fee_that_blocks_noise_traders_is_rejected(self):
        cfg = traders.TraderConfig(fee=0.05, noise_slippage=0.05)
        with self.assertRaises(ValueError):
            traders.simulate_day(LMSR([0.5, 0.5], 100), [0.5, 0.5], cfg, np.random.default_rng(1))

    def test_simulation_is_reproducible_and_keeps_bound(self):
        priors = [0.1, 0.2, 0.2, 0.2, 0.2, 0.1]
        cfg = traders.TraderConfig(informed_share=0.5, fee=0.01)
        runs = []
        for _ in range(2):
            m = LMSR(priors, 300)
            state = traders.simulate_day(m, priors, cfg, np.random.default_rng(1))
            runs.append(state['revenue'])
            self.assertGreaterEqual(m.pnl(0), -m.loss_bound() - 1e-6)
        self.assertEqual(*runs)


class Metrics(unittest.TestCase):
    def state(self):
        return {'revenue': 10.0, 'fee_income': 1.0, 'exposure': [20.0, 5.0], 'prices': [0.6, 0.4],
                'offered': 100.0, 'filled': 25.0, 'fills': 2, 'peak_worst_loss': 9.0}

    def test_pnl_uses_the_winning_exposure(self):
        self.assertEqual(mm_metrics.day_metrics(self.state(), 0, [0.5, 0.5])['pnl'], -9.0)
        self.assertEqual(mm_metrics.day_metrics(self.state(), 1, [0.5, 0.5])['pnl'], 6.0)

    def test_summary_uses_pooled_pnl_per_share(self):
        days = [mm_metrics.day_metrics(self.state(), w, [0.5, 0.5]) for w in (0, 1)]
        self.assertAlmostEqual(mm_metrics.summarize(days)['pnl_per_share'], -3 / 50)

    def test_summary_skips_labels_and_partial_metrics(self):
        first = {'day': '2026-05-01', 'split': 'test', **mm_metrics.day_metrics(self.state(), 0, [0.5, 0.5])}
        second = {'day': '2026-05-02', 'split': 'test', **mm_metrics.day_metrics(self.state(), 1, [0.5, 0.5], [0.6, 0.4])}
        s = mm_metrics.summarize([first, second])
        self.assertNotIn('day', s)
        self.assertEqual(s['tracking_error'], 0.0)

    def test_bootstrap_is_seeded_and_brackets_mean(self):
        i = mm_metrics.bootstrap_mean([1.0, 2.0, 3.0, 4.0])
        self.assertLess(i['ci_low'], i['mean'])
        self.assertGreater(i['ci_high'], i['mean'])
        self.assertEqual(i, mm_metrics.bootstrap_mean([1.0, 2.0, 3.0, 4.0]))


if __name__ == '__main__':
    unittest.main()
