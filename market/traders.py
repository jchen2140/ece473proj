"""Synthetic trader population for controlled what-if runs against an LMSR.

Real-flow replay (market/backtest.py) shows what happened; this varies who shows up. Informed beliefs
center on the later Kalshi midpoint, never the outcome: it stands in for traders who know where
the market is heading before the LMSR does. market/calibrate_traders.py fits the settings on train days.
"""
from dataclasses import dataclass

import numpy as np

SIDES = ('yes', 'no')


@dataclass
class TraderConfig:
    arrivals: int = 200
    informed_share: float = 0.25
    threshold: float = 0.03        # Belief traders need this much edge per share.
    belief_concentration: float = 200.0  # Higher means beliefs closer to the source.
    median_size: float = 50.0
    size_sigma: float = 1.0
    noise_slippage: float = 0.05   # Noise traders pay up to this much over the current price.
    fee: float = 0.0


def noisy_belief(source, concentration, rng):
    """Private belief: a Dirichlet draw centered on the source forecast."""
    return rng.dirichlet(np.asarray(source, dtype=float) * concentration)


def belief_order(belief, prices, threshold):
    """Most mispriced (outcome, side, limit), or None if nothing clears the threshold."""
    gaps = np.asarray(belief) - np.asarray(prices)
    outcome = int(np.abs(gaps).argmax())
    if abs(gaps[outcome]) <= threshold:
        return None
    if gaps[outcome] > 0:
        return outcome, 'yes', float(belief[outcome]) - threshold
    return outcome, 'no', 1 - float(belief[outcome]) - threshold


def noise_order(prices, slippage, rng):
    """Liquidity-driven order, more likely on bins near the money, independent of the outcome."""
    outcome = int(rng.choice(len(prices), p=prices))
    side = SIDES[int(rng.integers(2))]
    price = float(prices[outcome]) if side == 'yes' else 1 - float(prices[outcome])
    return outcome, side, min(0.99, price + slippage)


def simulate_day(market, belief_source, cfg, rng):
    """Run one day of arrivals through market in place and return the state for mm_metrics."""
    if cfg.fee >= cfg.noise_slippage:
        raise ValueError('Noise traders cannot fill when the fee absorbs all their slippage')
    n_informed = int(round(cfg.arrivals * cfg.informed_share))
    kinds = np.array([True] * n_informed + [False] * (cfg.arrivals - n_informed))
    rng.shuffle(kinds)
    state = {'fee_income': 0.0, 'offered': 0.0, 'filled': 0.0, 'fills': 0, 'peak_worst_loss': 0.0}
    for informed in kinds:
        prices = market.prices()
        if informed:
            order = belief_order(noisy_belief(belief_source, cfg.belief_concentration, rng), prices, cfg.threshold)
        else:
            order = noise_order(prices, cfg.noise_slippage, rng)
        size = float(rng.lognormal(np.log(cfg.median_size), cfg.size_sigma))
        state['offered'] += size
        if order is None:
            continue
        outcome, side, limit = order
        shares, _ = market.fill_to_limit(outcome, side, limit - cfg.fee, size)
        if shares:
            state['filled'] += shares
            state['fee_income'] += shares * cfg.fee
            state['fills'] += 1
            loss = market.worst_case_loss() - state['fee_income']
            state['peak_worst_loss'] = max(state['peak_worst_loss'], loss)
    state.update(revenue=market.revenue, exposure=market.exposure(), prices=market.prices())
    return state
