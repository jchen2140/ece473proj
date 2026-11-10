"""Checks for market/mechanism.py: its price paths and delayed tilt agree with market/makers.py."""
import unittest

import numpy as np

from market import makers, mechanism
from tests.test_makers import synthetic_day


class PricePathTest(unittest.TestCase):
    def test_final_snapshot_matches_the_replay(self):
        rng = np.random.default_rng(7)
        for trial in range(30):
            day = synthetic_day(rng, int(rng.integers(6)))
            for b in (30.0, 300.0):
                final = mechanism.price_path(day, day['priors']['public'], b, 0.03, [day['flow'][-1][0] + 1])[0]
                replayed = makers.replay(day, 'forecast', b, 0.03, 1.0, 2.0)['closing_brier']
                self.assertAlmostEqual(mechanism.mm_metrics.brier(final, day['winner']), replayed, places=10,
                                       msg=f'trial {trial}, b {b}')

    def test_snapshot_before_any_trade_is_the_prior(self):
        day = synthetic_day(np.random.default_rng(8), 2)
        first = mechanism.price_path(day, day['priors']['public'], 100.0, 0.0, [day['flow'][0][0] - 1])[0]
        np.testing.assert_allclose(first, day['priors']['public'] / day['priors']['public'].sum())

    def test_fixed_price_quoter_does_not_move(self):
        day = synthetic_day(np.random.default_rng(9), 3)
        final = mechanism.price_path(day, day['priors']['public'], mechanism.FIXED_B, 0.0, [day['flow'][-1][0] + 1])[0]
        np.testing.assert_allclose(final, day['priors']['public'], atol=1e-3)


class DelayedTiltTest(unittest.TestCase):
    def test_default_delay_matches_makers_tilt(self):
        day = synthetic_day(np.random.default_rng(10), 5)
        fn = mechanism.delayed_tilt(makers.FEED_DELAY)
        for ts in day['minute_ts'][::97]:
            a, ma = fn(day, ts)
            b, mb = makers.tilt(day, ts)
            np.testing.assert_array_equal(a, b)
            self.assertEqual(ma, mb)


if __name__ == '__main__':
    unittest.main()
