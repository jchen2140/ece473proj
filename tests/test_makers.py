"""Checks for market/makers.py: the live-truncation loss bound in docs/paper.md, and the tilt itself."""
import math
import unittest

import numpy as np

from common import DATA
from market import makers


def synthetic_day(rng, winner):
    """Six bins on 80-91F, a random prior, random aggressive flow, and a running maximum that never rules out the winner."""
    lows = np.array([-math.inf, 81.5, 83.5, 85.5, 87.5, 89.5])
    highs = np.array([81.5, 83.5, 85.5, 87.5, 89.5, math.inf])
    prior = rng.dirichlet(np.ones(6)) * 0.94 + 0.01
    prior /= prior.sum()
    start = 1_000_000.0
    flow = []
    for ts in np.sort(rng.uniform(start - 12 * 3600, start + 20 * 3600, 400)):
        is_yes = bool(rng.random() < 0.5)
        flow.append((float(ts), int(rng.integers(6)), is_yes, float(rng.uniform(0.02, 0.98)), float(rng.uniform(1, 200))))
    # The maximum climbs to just inside the winning bin (touching its lower edge at most), never past its upper edge.
    ceiling = lows[winner] + 0.4 if math.isfinite(lows[winner]) else 78.0
    minute_ts = start + np.arange(0, 86400, 60, dtype=float)
    running_max = np.maximum.accumulate(np.minimum(70 + 25 * rng.random(len(minute_ts)).cumsum() / len(minute_ts), ceiling))
    return {'priors': {'public': prior}, 'sd': 2.0, 'lows': lows, 'highs': highs, 'winner': winner, 'start': start,
            'flow': flow, 'minute_ts': minute_ts, 'running_max': running_max, 'reports': np.array([])}


class LiveTruncationBoundTest(unittest.TestCase):
    def test_loss_never_exceeds_the_bound(self):
        """Loss <= b ln(1/prior_w) + b ln(1/tau_w): truncating ruled-out bins never adds to the worst case."""
        rng = np.random.default_rng(473)
        for trial in range(200):
            winner = int(rng.integers(6))
            day = synthetic_day(rng, winner)
            for b in (30.0, 300.0):
                loss = -makers.replay(day, 'forecast_live', b, 0.0, 1.0, 2.0)['pnl']
                tau_w, _ = makers.tilt(day, day['flow'][-1][0])
                bound = b * math.log(1 / day['priors']['public'][winner]) + b * math.log(1 / tau_w[winner])
                self.assertLessEqual(loss, bound + 1e-6, f'trial {trial}, b {b}')

    def test_winner_is_never_written_off(self):
        rng = np.random.default_rng(1)
        for winner in range(6):
            day = synthetic_day(rng, winner)
            weights, _ = makers.tilt(day, day['minute_ts'][-1] + makers.FEED_DELAY)
            self.assertGreaterEqual(weights[winner], makers.TOUCHED_WEIGHT)


class TiltTest(unittest.TestCase):
    def test_bins_cleared_by_a_degree_are_written_off(self):
        day = synthetic_day(np.random.default_rng(2), 5)
        day['running_max'] = np.full(len(day['minute_ts']), 86.0)
        weights, m = makers.tilt(day, day['minute_ts'][10] + makers.FEED_DELAY)
        self.assertEqual(m, 86.0)
        np.testing.assert_allclose(weights, [makers.DEAD_WEIGHT, makers.DEAD_WEIGHT, makers.TOUCHED_WEIGHT, 1, 1, 1])

    def test_no_tilt_before_the_feed_delay(self):
        day = synthetic_day(np.random.default_rng(3), 5)
        weights, m = makers.tilt(day, day['minute_ts'][0])
        self.assertIsNone(m)
        self.assertTrue((weights == 1).all())


@unittest.skipUnless((DATA / 'raw' / 'realtime' / 'MSP' / 'trades.csv.gz').exists(), 'needs collect.realtime data')
class DataSmokeTest(unittest.TestCase):
    def test_day_inputs_loads_a_city(self):
        days = makers.day_inputs('MSP')
        self.assertGreater(len(days), 100)
        for d in days[:5]:
            for prior in d['priors'].values():
                self.assertAlmostEqual(float(prior.sum()), 1.0, places=6)


if __name__ == '__main__':
    unittest.main()
