"""Figures from backtest.py outputs; run after backtest.py. Writes data/results/backtest/figures/."""
import csv
import json

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import backtest
import forecasts

FIGURES = backtest.OUT / 'figures'
INK, MUTED, GRID, SURFACE = '#0b0b0b', '#52514e', '#e7e6e1', '#fcfcfb'
COLOR = dict(zip(forecasts.SEEDS, ('#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4')))
MARKER = dict(zip(forecasts.SEEDS, 'osD^v'))  # Identity is never carried by color alone.
FEE = 0.05
BMAX = 3000  # Larger b losses would flatten the region where the choice matters.


def read(name):
    with (backtest.OUT / name).open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def axes(title, xlabel, ylabel, size=(7.2, 4.2)):
    fig, ax = plt.subplots(figsize=size, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc='left', color=INK, fontsize=12, pad=12)
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(axis='y', color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, length=0)
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color(GRID)
    return fig, ax


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIGURES / name, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def pnl_by_seed(summary):
    table = summary['comparisons']['test/same_settings']['seeds']
    b, fee = table['noaa_calibrated']['settings']
    fig, ax = axes(f'Test-day PnL by opening forecast (all at b {b}, fee {fee*100:.0f} cents)', 'Mean daily PnL, dollars (95% CI)', '')
    ax.grid(axis='y', visible=False)
    ax.grid(axis='x', color=GRID, linewidth=0.8)
    order = sorted(forecasts.SEEDS, key=lambda s: table[s]['pnl']['mean'])
    for y, seed in enumerate(order):
        i = table[seed]['pnl']
        ax.plot([i['ci_low'], i['ci_high']], [y, y], color=COLOR[seed], linewidth=2, solid_capstyle='round')
        ax.scatter(i['mean'], y, s=70, color=COLOR[seed], marker=MARKER[seed], edgecolor=SURFACE, linewidth=1.5, zorder=3)
    ax.set_yticks(range(len(order)), [forecasts.LABEL[s] for s in order], color=INK)
    ax.axvline(0, color=MUTED, linewidth=1)
    save(fig, 'pnl_by_seed.png')


def pnl_by_b(grid):
    fig, ax = axes(f'Liquidity tradeoff on train days (fee {FEE*100:.0f} cents)', 'Liquidity parameter b (log scale)', 'Mean daily PnL, dollars')
    rows = [r for r in grid if r['stop'] == backtest.MAIN_STOP and r['split'] == 'train'
            and float(r['fee']) == FEE and float(r['b']) <= BMAX]
    for seed in forecasts.SEEDS:
        line = sorted((float(r['b']), float(r['mean_pnl'])) for r in rows if r['seed'] == seed)
        ax.plot(*zip(*line), color=COLOR[seed], marker=MARKER[seed], markersize=8, markeredgecolor=SURFACE,
                linewidth=2, label=forecasts.LABEL[seed])
    ax.set_xscale('log')
    ax.axhline(0, color=MUTED, linewidth=1)
    ax.legend(frameon=False, labelcolor=INK)
    save(fig, 'pnl_by_b.png')


def pnl_by_informed(synthetic):
    fig, ax = axes('Synthetic traders: more informed flow on test days', 'Informed share of arrivals', 'Mean daily PnL, dollars')
    rows = [r for r in synthetic if r['split'] == 'test']
    for seed in forecasts.SEEDS:
        line = sorted((float(r['informed_share']), float(r['mean_pnl'])) for r in rows if r['seed'] == seed)
        ax.plot(*zip(*line), color=COLOR[seed], marker=MARKER[seed], markersize=8, markeredgecolor=SURFACE,
                linewidth=2, label=forecasts.LABEL[seed])
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    ax.axhline(0, color=MUTED, linewidth=1)
    ax.legend(frameon=False, labelcolor=INK, loc='center left', bbox_to_anchor=(1, 0.5))
    save(fig, 'pnl_by_informed_share.png')


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    pnl_by_seed(json.loads((backtest.OUT / 'summary.json').read_text(encoding='utf-8')))
    pnl_by_b(read('replay_grid.csv'))
    pnl_by_informed(read('synthetic.csv'))
    print('Wrote', FIGURES)


if __name__ == '__main__':
    main()
