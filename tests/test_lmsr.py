import math
import unittest

import numpy as np

from market.lmsr import LMSR


class Pricing(unittest.TestCase):
    def test_opening_prices_equal_priors(self):
        p = [0.05, 0.1, 0.2, 0.3, 0.25, 0.1]
        self.assertTrue(np.allclose(LMSR(p, 500).prices(), p))

    def test_zero_prior_is_rejected(self):
        with self.assertRaises(ValueError):
            LMSR([0.0, 0.5, 0.5], 100)

    def test_buying_yes_raises_price_and_prices_stay_normalized(self):
        m = LMSR([0.25] * 4, 100)
        m.buy(1, 'yes', 50)
        self.assertGreater(m.prices()[1], 0.25)
        self.assertAlmostEqual(m.prices().sum(), 1)

    def test_no_shares_cost_matches_complement_identity(self):
        # d NO shares on i equals d guaranteed dollars minus d YES shares on i.
        a, b = LMSR([0.2, 0.3, 0.5], 100), LMSR([0.2, 0.3, 0.5], 100)
        paid = a.buy(0, 'no', 40)
        self.assertAlmostEqual(paid, 40 + b.trade_cost([-40, 0, 0]))
        self.assertTrue(np.allclose(a.prices(), b.prices(b.q + [-40, 0, 0])))

    def test_round_trip_costs_nothing(self):
        m = LMSR([0.2, 0.3, 0.5], 100)
        paid = m.buy(2, 'yes', 30)
        self.assertAlmostEqual(paid + m.trade_cost([0, 0, -30]), 0)

    def test_extreme_position_is_numerically_stable(self):
        m = LMSR([1 / 6] * 6, 1)
        m.buy(0, 'yes', 5000)
        self.assertTrue(np.isfinite(m.prices()).all())
        self.assertAlmostEqual(m.prices()[0], 1)


class Fills(unittest.TestCase):
    def test_limit_fill_stops_at_limit(self):
        m = LMSR([0.2, 0.3, 0.5], 100)
        shares, paid = m.fill_to_limit(0, 'yes', 0.4, 10_000)
        self.assertAlmostEqual(m.prices()[0], 0.4)
        self.assertLess(paid / shares, 0.4)

    def test_limit_fill_for_no_side(self):
        m = LMSR([0.2, 0.3, 0.5], 100)
        m.fill_to_limit(2, 'no', 0.7, 10_000)
        self.assertAlmostEqual(1 - m.prices()[2], 0.7)

    def test_limit_fill_respects_size(self):
        m = LMSR([0.2, 0.3, 0.5], 100)
        shares, _ = m.fill_to_limit(0, 'yes', 0.9, 3)
        self.assertEqual(shares, 3)

    def test_no_fill_when_price_is_worse_than_limit(self):
        m = LMSR([0.2, 0.3, 0.5], 100)
        self.assertEqual(m.fill_to_limit(2, 'yes', 0.4, 100), (0.0, 0.0))
        self.assertEqual(m.fill_to_limit(2, 'no', 0.4, 100), (0.0, 0.0))


class Settlement(unittest.TestCase):
    def test_loss_never_exceeds_bound(self):
        p = [0.05, 0.15, 0.3, 0.3, 0.15, 0.05]
        rng = np.random.default_rng(1)
        for _ in range(200):
            m = LMSR(p, 100)
            for _ in range(30):
                m.buy(int(rng.integers(6)), 'yes' if rng.random() < 0.5 else 'no', float(rng.uniform(0, 80)))
            for winner in range(6):
                self.assertGreaterEqual(m.pnl(winner), -100 * math.log(1 / 0.05) - 1e-6)

    def test_all_in_on_winner_reaches_bound_for_that_outcome(self):
        m = LMSR([0.1, 0.9], 100)
        m.buy(0, 'yes', 5000)
        self.assertAlmostEqual(m.pnl(0), -100 * math.log(10), places=6)

    def test_pnl_is_revenue_minus_winning_shares(self):
        m = LMSR([0.5, 0.5], 100)
        paid = m.buy(0, 'yes', 10)
        self.assertAlmostEqual(m.pnl(0), paid - 10)
        self.assertAlmostEqual(m.pnl(1), paid)

    def test_worst_case_loss_matches_worst_outcome(self):
        m = LMSR([0.2, 0.3, 0.5], 100)
        m.buy(1, 'yes', 20)
        m.buy(0, 'no', 5)
        self.assertAlmostEqual(m.worst_case_loss(), -min(m.pnl(i) for i in range(3)))


if __name__ == '__main__':
    unittest.main()
