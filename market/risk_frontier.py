"""Risk and capital efficiency of the liquidity parameter b, from the full-year replay grid.

An LMSR with prior pi that sells shares q loses at most b * ln(1 / pi_w) if outcome w wins (the
standard bound; docs/paper.md Section 6.4 extends it to live truncation). Two capital measures follow:

- Worst-case capital, b * ln(1 / min_i pi_i): what the maker must hold to survive any outcome.
- Expected capital, the average of b * ln(1 / pi_w) over the realized winners. That average is exactly
  b times the prior's multiclass log loss, so a forecast that lowers log loss by x lowers the expected
  subsidy by b * x per day at the same liquidity.

A sharp forecast lowers expected capital but can raise worst-case capital (its smallest bin
probability is smaller than a uniform prior's 1/6). This module measures both, and the realized tail
(worst day, 5th-percentile day, peak worst-case loss), as b moves with the fee fixed. It then compares
makers at equal capital by interpolating mean PnL against capital along each maker's b curve.

Run after market/full_year.py. Writes data/results/risk_frontier/. Each city's walk-forward test months only.
"""
import sys

import numpy as np
import pandas as pd

import cities
from common import DATA, write_csv
from market import full_year
from market import metrics as mm_metrics

OUT = DATA / 'results' / 'risk_frontier'
MAKERS = ('generic', 'forecast', 'forecast_live', 'forecast_live_cons', 'forecast_live_filtered', 'kalshi')
BUDGETS = (50, 150, 500)  # Expected-capital budgets ($/day) for the equal-capital comparison.


def frontier(grid, margins):
    """margins: test month -> the conservative margin that month actually used."""
    # The walk-forward test months of this city (they start later where the margin needs more days).
    test = grid[(grid['fee'] == full_year.FRONTIER_FEE) & grid['day'].str[:7].isin(set(margins))
                & grid['maker'].isin(MAKERS)]
    # The conservative maker has one replay per monthly margin; keep the margin each month actually used.
    test = test[(test['maker'] != 'forecast_live_cons') | (test['margin_f'] == test['day'].str[:7].map(margins.get))]
    rows = []
    for (code, maker, b), g in test.groupby(['city', 'maker', 'b']):
        pnl = g['pnl'].to_numpy()
        ci = mm_metrics.bootstrap_mean(pnl)
        rows.append({'city': code, 'maker': maker, 'b': b, 'days': len(g), 'mean_pnl': pnl.mean(), 'ci_low': ci['ci_low'],
                     'ci_high': ci['ci_high'], 'worst_day': pnl.min(), 'p05_pnl': np.percentile(pnl, 5),
                     'peak_worst_case_loss': g['peak_worst_case_loss'].mean(),
                     'expected_capital': g['opening_bound'].mean(), 'worst_case_capital': g['max_opening_bound'].mean(),
                     'prior_log_loss': g['opening_bound'].mean() / b,
                     'return_on_expected_capital': pnl.mean() / g['opening_bound'].mean(),
                     'return_on_worst_case_capital': pnl.mean() / g['max_opening_bound'].mean(),
                     'closing_brier': g['closing_brier'].mean()})
    return rows


def equal_capital(rows):
    """Mean PnL each maker earns at a fixed expected-capital budget, interpolated on log capital."""
    d = pd.DataFrame(rows)
    out = []
    for (code, maker), g in d.groupby(['city', 'maker']):
        g = g.sort_values('expected_capital')
        x = np.log(g['expected_capital'].to_numpy())
        for budget in BUDGETS:
            inside = x[0] <= np.log(budget) <= x[-1]
            out.append({'city': code, 'maker': maker, 'budget': budget,
                        'mean_pnl': float(np.interp(np.log(budget), x, g['mean_pnl'])) if inside else np.nan,
                        'implied_b': float(budget / g['prior_log_loss'].mean())})
    return out


def plot(rows, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    d = pd.DataFrame(rows)
    codes = [c for c in cities.FULL_YEAR_CITIES if c in set(d['city'])]
    colors = {'generic': '#9aa5b1', 'forecast': '#2a78d6', 'forecast_live': '#6250d6', 'forecast_live_cons': '#eb6834', 'forecast_live_filtered': '#c0398a',
              'kalshi': '#1baf7a'}
    fig, axes = plt.subplots(2, (len(codes) + 1) // 2, figsize=(3.2 * ((len(codes) + 1) // 2), 6.2), squeeze=False)
    for ax, code in zip(axes.flat, codes):
        for maker in MAKERS:
            g = d[(d['city'] == code) & (d['maker'] == maker)].sort_values('b')
            if g.empty:
                continue
            ax.plot(g['worst_case_capital'], g['mean_pnl'], marker='o', ms=4, lw=1.8, color=colors[maker],
                    label=full_year.LABEL[maker].split(' (')[0])
        ax.axhline(0, color='#52514e', lw=0.6)
        ax.set_xscale('log')
        ax.set_title(code, loc='left', fontsize=9)
        ax.tick_params(labelsize=7)
        ax.spines[['top', 'right']].set_visible(False)
    for ax in axes.flat[len(codes):]:
        ax.axis('off')
    for ax in axes[:, 0]:
        ax.set_ylabel('Mean test-day PnL ($)', fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel('Worst-case capital b·ln(1/min π) ($, log)', fontsize=8)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower right', frameon=False, fontsize=8)
    fig.suptitle(f'PnL against worst-case capital as b varies (fee ${full_year.FRONTIER_FEE:.2f}, test months)',
                 x=0.01, ha='left', fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def money(v):
    return '' if pd.isna(v) else (f'-${abs(v):,.0f}' if v < 0 else f'${v:,.0f}')


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    (OUT / 'figures').mkdir(parents=True, exist_ok=True)
    grid = pd.read_csv(full_year.OUT / 'grid_daily.csv.gz')
    settings = pd.read_csv(full_year.OUT / 'settings.csv')
    cons = settings[settings['maker'] == 'forecast_live_cons']
    rows = []
    for code in grid['city'].unique():
        rows += frontier(grid[grid['city'] == code], cons[cons['city'] == code].set_index('month')['margin_f'].to_dict())
    equal = equal_capital(rows)
    write_csv('results/risk_frontier/frontier.csv', rows)
    write_csv('results/risk_frontier/equal_capital.csv', equal)
    plot(rows, OUT / 'figures' / 'frontier.png')
    d = pd.DataFrame(rows)
    codes = [c for c in cities.FULL_YEAR_CITIES if c in set(d['city'])]
    lines = ['# Risk and capital efficiency of b (generated by market/risk_frontier.py; do not edit by hand)', '',
             f'Full-year replay grid, each city\'s walk-forward test months only, fee fixed at ${full_year.FRONTIER_FEE:.2f}. '
             'Expected capital = mean of b·ln(1/π_winner) = b × the opening prior\'s log loss; worst-case capital = '
             'mean of b·ln(1/min π).', '',
             '## 1. Opening-prior log loss (capital per unit of b)', '',
             '| Maker | ' + ' | '.join(codes) + ' |', '| --- |' + ' ---: |' * len(codes)]
    for maker in MAKERS:
        lines.append(f'| {full_year.LABEL[maker]} | ' + ' | '.join(
            f"{d[(d['city'] == c) & (d['maker'] == maker)]['prior_log_loss'].mean():.3f}" for c in codes) + ' |')
    lines += ['', '## 2. Mean PnL at equal expected capital (interpolated along b)', '',
              '| City | Budget ($/day) | ' + ' | '.join(full_year.LABEL[m].split(' (')[0] for m in MAKERS) + ' |',
              '| --- | ---: |' + ' ---: |' * len(MAKERS)]
    e = pd.DataFrame(equal)
    for code in codes:
        for budget in BUDGETS:
            cells = [money(e[(e['city'] == code) & (e['maker'] == m) & (e['budget'] == budget)]['mean_pnl'].squeeze()) for m in MAKERS]
            lines.append(f'| {code} | {budget} | ' + ' | '.join(cells) + ' |')
    lines += ['', '## 3. Full frontier', '',
              '| City | Maker | b | Mean PnL [95% CI] | 5th pct day | Worst day | Peak worst-case loss | Expected capital | '
              'Worst-case capital | Return on worst-case capital |',
              '| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in rows:
        lines.append(f"| {r['city']} | {full_year.LABEL[r['maker']]} | {r['b']:g} | {money(r['mean_pnl'])} "
                     f"[{money(r['ci_low'])}, {money(r['ci_high'])}] | {money(r['p05_pnl'])} | {money(r['worst_day'])} | "
                     f"{money(r['peak_worst_case_loss'])} | {money(r['expected_capital'])} | {money(r['worst_case_capital'])} | "
                     f"{r['return_on_worst_case_capital']:+.3f} |")
    text = '\n'.join(lines) + '\n'
    (OUT / 'risk_frontier_tables.md').write_text(text, encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
