"""Logarithmic market scoring rule for one event with mutually exclusive outcomes.

Minimal maker used by the backtest. The maker's position is the vector q of outstanding YES
shares per outcome; seeding with priors p sets q0 = b*ln(p), so the opening prices equal p.
Buying NO on outcome i is the same as buying one YES share of every other outcome.
"""
import math

import numpy as np


class LMSR:
    def __init__(self, priors, b):
        p = np.asarray(priors, dtype=float)
        if b <= 0:
            raise ValueError('b must be positive')
        if len(p) < 2 or np.any(p <= 0) or not math.isclose(float(p.sum()), 1, abs_tol=1e-9):
            raise ValueError('Priors must be positive and sum to one')
        self.b = float(b)
        self.q = self.b * np.log(p)
        self.q0 = self.q.copy()
        self.opening_cost = self.cost(self.q)
        self.revenue = 0.0

    def cost(self, q):
        scaled = np.asarray(q, dtype=float) / self.b
        top = scaled.max()
        return float(self.b * (top + np.log(np.exp(scaled - top).sum())))  # Stable log-sum-exp.

    def prices(self, q=None):
        scaled = (self.q if q is None else np.asarray(q, dtype=float)) / self.b
        weights = np.exp(scaled - scaled.max())
        return weights / weights.sum()

    def trade_cost(self, delta):
        """Cash the maker receives for selling delta shares (negative means it pays out)."""
        return self.cost(self.q + np.asarray(delta, dtype=float)) - self.cost(self.q)

    @staticmethod
    def yes_delta(outcome, shares, n):
        delta = np.zeros(n)
        delta[outcome] = shares
        return delta

    @staticmethod
    def no_delta(outcome, shares, n):
        delta = np.full(n, float(shares))
        delta[outcome] = 0.0
        return delta

    def delta(self, outcome, side, shares):
        builder = self.yes_delta if side == 'yes' else self.no_delta
        return builder(outcome, shares, len(self.q))

    def side_price(self, outcome, side):
        p = float(self.prices()[outcome])
        return p if side == 'yes' else 1 - p

    def buy(self, outcome, side, shares):
        """Sell shares to a taker and return the cash paid."""
        if shares < 0:
            raise ValueError('shares must be nonnegative')
        delta = self.delta(outcome, side, shares)
        paid = self.trade_cost(delta)
        self.q = self.q + delta
        self.revenue += paid
        return paid

    def shares_to_limit(self, outcome, side, limit):
        """Shares a taker can buy before the marginal price of its side reaches limit."""
        if not 0 < limit < 1:
            return 0.0
        p = float(self.prices()[outcome])
        target_yes = limit if side == 'yes' else 1 - limit
        if (side == 'yes' and p >= target_yes) or (side == 'no' and p <= target_yes):
            return 0.0
        # Moving q_i alone by x gives odds(p') = odds(p) * exp(x/b).
        x = self.b * (math.log(target_yes / (1 - target_yes)) - math.log(p / (1 - p)))
        return abs(x)

    def fill_to_limit(self, outcome, side, limit, max_shares):
        """Fill up to max_shares while the marginal price stays at or better than limit."""
        shares = min(float(max_shares), self.shares_to_limit(outcome, side, limit))
        if shares <= 0:
            return 0.0, 0.0
        return shares, self.buy(outcome, side, shares)

    def exposure(self):
        """Shares owed per outcome net of the opening position."""
        return self.q - self.q0

    def payout(self, winner):
        return float(self.exposure()[winner])

    def pnl(self, winner):
        return self.revenue - self.payout(winner)

    def worst_case_loss(self):
        """Largest settlement loss across outcomes at the current position."""
        return float(self.exposure().max() - self.revenue)

    def loss_bound(self, priors=None):
        """Theoretical bound b*ln(1/min p) on the loss from the opening prices."""
        p = self.prices(self.q0) if priors is None else np.asarray(priors, dtype=float)
        return self.b * math.log(1 / float(p.min()))
