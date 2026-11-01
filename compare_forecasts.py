"""Calibrated NOAA versus Kalshi: accuracy, calibration, and whether disagreement carries signal.

Run after calibrate.py and evaluate.py. Test days are the headline; the blend is fit on train
days only. Disagreement is NOAA minus the noon Kalshi midpoint, per contract. Intervals resample
whole days. Writes data/results/forecast_comparison/.
"""
import csv
import math
import sys
from collections import defaultdict

import numpy as np

import backtest
import config
import evaluate
import forecasts
import plot_results
from collect_nyc import DATA, write_csv

OUT = DATA / 'results' / 'forecast_comparison'
MODELS = ['noaa_calibrated', 'raw_ensemble', 'climatology', 'market']
evaluate.COLUMN['noaa_calibrated'] = 'noaa_calibrated_probability'
evaluate.LABEL['noaa_calibrated'] = 'Calibrated NOAA'
evaluate.COLUMN['blend'] = 'blend_probability'
evaluate.LABEL['blend'] = 'Kalshi + NOAA blend'
PAIRS = [('noaa_calibrated', 'market'), ('noaa_calibrated', 'raw_ensemble'),
         ('noaa_calibrated', 'climatology'), ('blend', 'market')]
BUCKETS = ((0.0, 0.05), (0.05, 0.15), (0.15, 1.01))
CLIP = 0.005


def logit(p):
    p = np.clip(p, CLIP, 1 - CLIP)
    return np.log(p / (1 - p))


def fit_logistic(x, y, steps=30):
    """Newton steps for logistic regression with a small ridge; x includes the intercept."""
    beta = np.zeros(x.shape[1])
    for _ in range(steps):
        p = 1 / (1 + np.exp(-x @ beta))
        weight = np.maximum(p * (1 - p), 1e-6)
        hessian = (x * weight[:, None]).T @ x + 1e-6 * np.eye(x.shape[1])
        beta += np.linalg.solve(hessian, x.T @ (y - p))
    return beta


def design(rows):
    return np.column_stack([np.ones(len(rows)), logit(np.array([float(r['market_midpoint']) for r in rows])),
                            logit(np.array([float(r['noaa_calibrated_probability']) for r in rows]))])


def slope(x, y):
    x = x - x.mean()
    return float((x * (y - y.mean())).sum() / (x * x).sum())


def day_bootstrap(rows, statistic, replicates=config.BOOTSTRAP_REPLICATES, seed=config.BOOTSTRAP_SEED):
    """Statistic with an interval from resampling whole days."""
    groups = defaultdict(list)
    for r in rows:
        groups[r['target_date']].append(r)
    days = sorted(groups)
    rng = np.random.default_rng(seed)
    draws = [statistic([r for i in rng.integers(0, len(days), len(days)) for r in groups[days[i]]]) for _ in range(replicates)]
    return {'mean': statistic(rows), 'ci_low': float(np.percentile(draws, 2.5)), 'ci_high': float(np.percentile(draws, 97.5)),
            'n_days': len(days), 'n_contracts': len(rows)}


def signal_tests(rows):
    """Does NOAA minus Kalshi predict the outcome surprise, and the later Kalshi price move?"""
    vec = lambda rs, key: np.array([float(r[key]) for r in rs])
    outcome = lambda rs: slope(vec(rs, 'disagreement'), vec(rs, 'surprise'))
    move = lambda rs: slope(vec(rs, 'disagreement'), vec(rs, 'later_move'))
    moved = [r for r in rows if r['later_move'] != '']
    return {'outcome_surprise_slope': day_bootstrap(rows, outcome), 'later_move_slope': day_bootstrap(moved, move)}


def bucket_table(rows, split):
    out = []
    for low, high in BUCKETS:
        group = [r for r in rows if low <= abs(float(r['disagreement'])) < high]
        signed = lambda r: math.copysign(1, float(r['disagreement']))
        edge = lambda rs: float(np.mean([signed(r) * float(r['surprise']) for r in rs]))
        taker = lambda rs: float(np.mean([signed(r) * float(r['surprise']) - float(r['half_spread']) for r in rs]))
        a, b = day_bootstrap(group, edge), day_bootstrap(group, taker)
        out.append({'split': split, 'abs_disagreement_from': low, 'abs_disagreement_to': min(high, 1.0),
                    'contracts': len(group), 'days': a['n_days'], 'edge_at_midpoint': a['mean'],
                    'edge_ci_low': a['ci_low'], 'edge_ci_high': a['ci_high'],
                    'edge_after_half_spread': b['mean'], 'after_spread_ci_low': b['ci_low'], 'after_spread_ci_high': b['ci_high']})
    return out


def joined_rows():
    with (evaluate.RESULTS / 'scored_rows.csv').open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    calibrated = {r['market_ticker']: r['noaa_calibrated_probability'] for r in forecasts.read_rows('noaa_calibrated.csv')}
    records = forecasts.day_records()
    marks = backtest.load_kalshi_marks(records)
    later = {t: marks[r['day']][i] for r in records if r['day'] in marks for i, t in enumerate(r['tickers'])}
    out = []
    for r in rows:
        if r['market_ticker'] in calibrated and r['market_midpoint'] != '' and r['split']:
            mid = float(r['market_midpoint'])
            noaa = float(calibrated[r['market_ticker']])
            out.append({**r, 'noaa_calibrated_probability': noaa, 'disagreement': noaa - mid, 'surprise': int(r['outcome']) - mid,
                        'half_spread': (float(r['market_ask']) - float(r['market_bid'])) / 2,
                        'later_move': later[r['market_ticker']] - mid if r['market_ticker'] in later else ''})
    return out


def cell(i):
    return f"{i['mean']:.3f} [{i['ci_low']:.3f}, {i['ci_high']:.3f}]"


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    OUT.mkdir(parents=True, exist_ok=True)
    rows = joined_rows()
    train = [r for r in rows if r['split'] == 'train']
    test = [r for r in rows if r['split'] == 'test']
    beta = fit_logistic(design(train), np.array([int(r['outcome']) for r in train], dtype=float))
    for r, p in zip(rows, 1 / (1 + np.exp(-design(rows) @ beta))):
        r['blend_probability'] = float(p)
    models = MODELS + ['blend']
    result = evaluate.bootstrap(test, models, PAIRS)
    signal = {'train': signal_tests(train), 'test': signal_tests(test)}
    buckets = bucket_table(train, 'train') + bucket_table(test, 'test')
    write_csv('results/forecast_comparison/scores.csv', evaluate.results_rows(result, 'test'))
    write_csv('results/forecast_comparison/paired_differences.csv', evaluate.pair_rows(result, 'test'))
    write_csv('results/forecast_comparison/disagreement_buckets.csv', buckets)
    bins = {m: plot_results.reliability_bins(test, m) for m in models}
    write_csv('results/forecast_comparison/calibration.csv', [b for v in bins.values() for b in v])
    plot_results.plot_reliability(bins, 'Predicted probability vs realized YES rate (test days)', OUT / 'reliability.png',
                                  ['noaa_calibrated', 'raw_ensemble', 'market', 'blend'])
    lines = ['# NOAA versus Kalshi (generated by compare_forecasts.py; do not edit by hand)', '',
             evaluate.markdown(result, 'Accuracy on test days', 'Same contracts for every model. The blend was fit on train days only.'),
             f'Blend weights in log-odds: Kalshi {beta[1]:.3f}, NOAA {beta[2]:.3f}.', '',
             '### Does disagreement carry signal?', '',
             'A slope of 0 means no signal. For outcomes, 1 would mean moving Kalshi fully to NOAA. Later move is the Kalshi '
             'midpoint at noon on the target day minus its noon cutoff value.', '',
             '| Split | Outcome surprise on disagreement | Later Kalshi move on disagreement |', '| --- | --- | --- |']
    for split in ('train', 'test'):
        lines.append(f"| {split} | {cell(signal[split]['outcome_surprise_slope'])} | {cell(signal[split]['later_move_slope'])} |")
    lines += ['', '### Betting with NOAA by size of disagreement', '', 'Dollars per contract, trading in the direction of NOAA. Before fees.', '',
              '| Split | Disagreement | Contracts | Edge at midpoint [95% CI] | After half spread [95% CI] |', '| --- | --- | ---: | --- | --- |']
    for b in buckets:
        lines.append(f"| {b['split']} | {b['abs_disagreement_from']:.2f}-{b['abs_disagreement_to']:.2f} | {b['contracts']} | "
                     f"{b['edge_at_midpoint']:.3f} [{b['edge_ci_low']:.3f}, {b['edge_ci_high']:.3f}] | "
                     f"{b['edge_after_half_spread']:.3f} [{b['after_spread_ci_low']:.3f}, {b['after_spread_ci_high']:.3f}] |")
    text = '\n'.join(lines) + '\n'
    (OUT / 'comparison_tables.md').write_text(text, encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
