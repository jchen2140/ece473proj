"""Market-making evaluation metrics. PnL is in dollars per day for one six-bin event.

Per-day metrics come from one simulated maker. Comparisons resample whole days, as in
analysis/evaluate.py, because the fills and outcomes within a day are not independent.
"""
import config
import numpy as np


def brier(prices, winner):
    """Mean per binary contract, matching analysis/evaluate.py."""
    return float(np.mean([(p - (i == winner)) ** 2 for i, p in enumerate(prices)]))


def day_metrics(state, winner, opening, kalshi=None):
    """Summarize a finished day. state: revenue, fee_income, exposure, prices, offered, filled, fills, peak_worst_loss."""
    exposure = np.asarray(state['exposure'], dtype=float)
    payout = float(exposure[winner])
    pnl = state['revenue'] + state['fee_income'] - payout
    out = {'pnl': pnl, 'trading_pnl': state['revenue'] - payout, 'fee_income': state['fee_income'],
           'filled_shares': state['filled'], 'offered_shares': state['offered'],
           'fill_rate': state['filled'] / state['offered'] if state['offered'] else 0.0,
           'fills': state['fills'], 'cash_collected': state['revenue'] + state['fee_income'],
           'peak_worst_case_loss': state['peak_worst_loss'],
           'final_inventory': float(np.abs(exposure - exposure.mean()).max()),
           'opening_brier': brier(opening, winner), 'closing_brier': brier(state['prices'], winner)}
    out['pnl_per_share'] = pnl / state['filled'] if state['filled'] else 0.0
    if kalshi is not None:
        out['kalshi_brier'] = brier(kalshi, winner)
        out['tracking_error'] = float(np.mean(np.abs(np.asarray(state['prices']) - np.asarray(kalshi))))
    return out


def summarize(days):
    """Average day-level metrics, plus downside statistics of daily PnL."""
    pnl = np.array([d['pnl'] for d in days])
    keys = {k for d in days for k, v in d.items() if k != 'pnl' and not isinstance(v, str)}
    out = {k: float(np.mean([d[k] for d in days if k in d])) for k in sorted(keys)}
    filled = sum(d['filled_shares'] for d in days)
    out.update({'days': len(days), 'mean_pnl': float(pnl.mean()), 'median_pnl': float(np.median(pnl)),
                'sd_pnl': float(pnl.std(ddof=1)) if len(pnl) > 1 else 0.0,
                'worst_day': float(pnl.min()), 'p05_pnl': float(np.percentile(pnl, 5)),
                'share_days_profitable': float(np.mean(pnl > 0)),
                'pnl_per_share': float(pnl.sum() / filled) if filled else 0.0})
    out['daily_sharpe'] = out['mean_pnl'] / out['sd_pnl'] if out['sd_pnl'] else 0.0
    return out


def bootstrap_mean(values, replicates=config.BOOTSTRAP_REPLICATES, seed=config.BOOTSTRAP_SEED):
    """Day-level mean with a 95% percentile interval; pass paired differences for comparisons."""
    values = np.asarray(values, dtype=float)
    draws = np.random.default_rng(seed).integers(0, len(values), size=(replicates, len(values)))
    means = values[draws].mean(axis=1)
    return {'mean': float(values.mean()), 'ci_low': float(np.percentile(means, 2.5)),
            'ci_high': float(np.percentile(means, 97.5)), 'share_positive': float(np.mean(means > 0))}
