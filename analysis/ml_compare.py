"""ML forecast versus Kalshi and the NOAA forecasts: accuracy, where each wins, and tradable edge.

Run after analysis/evaluate.py, forecast/noaa_calibration.py and forecast/ml_model.py. The ML forecasts are walk-forward, so they
are out of sample on train days too; the blend and the trading threshold are still fit on train
days only and scored on test days. Writes data/results/ml/.

- Scores: Brier and log loss on identical test contracts, day-bootstrap intervals.
- Temperature accuracy: point-forecast errors, including the high Kalshi's prices imply.
- Regimes: per-day Brier difference (ML minus Kalshi) by season and weather type, all window days.
- Disagreement: does ML minus Kalshi predict outcomes or later price moves; trading at the
  quoted ask/bid with Kalshi's taker fee.
- Importance: grouped permutation importance, model fit before the test period.
"""
import json
import math
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from analysis import compare_forecasts as cf
import config
from analysis import evaluate
from forecast import probabilities as forecasts
from forecast import ml_model
from common import DATA, write_csv

OUT = ml_model.OUT
FEE_RATE = 0.07  # Kalshi taker fee per contract: 0.07 * P * (1 - P), rounded up to the cent per order.
THRESHOLDS = (0.02, 0.05, 0.08, 0.10, 0.15, 0.20)
for key, column, label in [('ml', 'ml_probability', ml_model.LABEL['ml']),
                           ('consensus', 'consensus_probability', ml_model.LABEL['consensus']),
                           ('noaa_calibrated', 'noaa_calibrated_probability', 'Calibrated NOAA'),
                           ('blend', 'blend_probability', 'Kalshi + ML blend')]:
    evaluate.COLUMN[key], evaluate.LABEL[key] = column, label
MODELS = ['ml', 'consensus', 'noaa_calibrated', 'raw_ensemble', 'climatology', 'market', 'blend']
PAIRS = [('ml', 'market'), ('ml', 'noaa_calibrated'), ('ml', 'consensus'), ('blend', 'market')]


def fee(price):
    return FEE_RATE * price * (1 - price)


def load_rows():
    rows = evaluate.read_csv(evaluate.RESULTS / 'scored_rows.csv')
    noaa = {r['market_ticker']: r['noaa_calibrated_probability'] for r in forecasts.read_rows('noaa_calibrated.csv')}
    ml = {r['market_ticker']: r for r in forecasts.read_rows('ml_forecasts.csv')}
    return [{**r, 'ml_probability': ml[r['market_ticker']]['ml_probability'],
             'consensus_probability': ml[r['market_ticker']]['consensus_probability'],
             'noaa_calibrated_probability': noaa[r['market_ticker']]}
            for r in rows if r['split'] and r['market_midpoint'] != '' and r['market_ticker'] in noaa and r['market_ticker'] in ml]


def blend(rows):
    """Logistic blend of Kalshi and ML in log-odds, fit on train days only."""
    design = lambda rs: np.column_stack([np.ones(len(rs)), cf.logit(np.array([float(r['market_midpoint']) for r in rs])),
                                         cf.logit(np.array([float(r['ml_probability']) for r in rs]))])
    train = [r for r in rows if r['split'] == 'train']
    beta = cf.fit_logistic(design(train), np.array([int(r['outcome']) for r in train], dtype=float))
    for r, p in zip(rows, 1 / (1 + np.exp(-design(rows) @ beta))):
        r['blend_probability'] = float(p)
    return beta


# ---------------------------------------------------------------- temperature view

def implied_normal(rows, probs):
    """Mean and sd of the normal whose bin probabilities best match `probs` (cross-entropy)."""
    probs = np.asarray(probs) / np.sum(probs)
    edges = [forecasts.bin_edges(r) for r in rows]
    inner = [(p, (lo + hi) / 2) for p, (lo, hi) in zip(probs, edges) if math.isfinite(lo) and math.isfinite(hi)]
    centre = sum(p * c for p, c in inner) / max(1e-9, sum(p for p, _ in inner))

    def loss(x):
        q = np.clip(forecasts.gaussian_bins(rows, x[0], math.exp(x[1])), 1e-9, 1)
        return -float(np.sum(probs * np.log(q)))

    best = minimize(loss, [centre, math.log(2.5)], method='Nelder-Mead', options={'xatol': 1e-3, 'fatol': 1e-6})
    return best.x[0], math.exp(best.x[1])


def temperature_table(rows, frame, ml_daily):
    """Point errors: each source, ML, and the high implied by Kalshi's noon prices."""
    groups = forecasts.day_groups()
    noaa = {r['target_date']: float(r['forecast_mean_f']) for r in forecasts.read_rows('noaa_calibrated.csv')}
    by_day = {}
    for r in rows:
        by_day.setdefault(r['target_date'], []).append(r)
    implied = {}
    for d, day_rows in by_day.items():
        mids = {r['market_ticker']: float(r['market_midpoint']) for r in day_rows}
        if len(mids) == len(groups[d]):
            implied[d] = implied_normal(groups[d], [mids[r['market_ticker']] for r in groups[d]])
    point = {'ECMWF IFS': 'ecmwf_max_f', 'NBM (MOS)': 'nbm_max_f', 'GFS MOS (MAV)': 'mav_max_f', 'NAM MOS (MET)': 'met_max_f',
             'Forecast consensus': 'baseline', 'GEFS mean (raw)': 'gefs_mean_f'}
    series = {label: {d: frame.at[pd.Timestamp(d), c] for d in by_day if pd.Timestamp(d) in frame.index} for label, c in point.items()}
    series['Calibrated NOAA'] = {d: noaa[d] for d in by_day if d in noaa}
    ml = ml_daily[ml_daily['model'] == 'ml'].set_index('target_date')['mean_f']
    series[ml_model.LABEL['ml']] = {d: ml[d] for d in by_day if d in ml.index}
    series['Kalshi implied (noon)'] = {d: v[0] for d, v in implied.items()}
    highs = {d: float(v[0]['kalshi_settlement_high_f']) for d, v in by_day.items()}
    table = []
    for split in ('train', 'test'):
        for label, values in series.items():
            errs = np.array([highs[d] - v for d, v in values.items() if config.split_of(d) == split and pd.notna(v)])
            table.append({'split': split, 'forecast': label, 'days': len(errs), 'mae_f': np.abs(errs).mean(),
                          'rmse_f': math.sqrt((errs ** 2).mean()), 'bias_f': errs.mean(),
                          'within_1f': float(np.mean(np.abs(errs) <= 1)), 'within_2f': float(np.mean(np.abs(errs) <= 2))})
    return table, implied


# ---------------------------------------------------------------- where ML wins or loses

def per_day_brier(rows, key):
    out = {}
    for r in rows:
        out.setdefault(r['target_date'], []).append((float(r[evaluate.COLUMN[key]]) - int(r['outcome'])) ** 2)
    return {d: float(np.mean(v)) for d, v in out.items()}


def regimes(rows, frame, implied, ml_daily):
    """Mean per-day Brier difference (ML minus Kalshi) by condition, on all window days."""
    ml, market = per_day_brier(rows, 'ml'), per_day_brier(rows, 'market')
    means = ml_daily[ml_daily['model'] == 'ml'].set_index('target_date')['mean_f']
    d = pd.DataFrame({'day': list(ml), 'diff': [ml[k] - market[k] for k in ml]})
    d['ts'] = pd.to_datetime(d['day'])
    f = frame.reindex(d['ts'])
    d['split'] = d['day'].map(config.split_of)
    season = {12: 'winter', 1: 'winter', 2: 'winter', 3: 'spring', 4: 'spring', 5: 'spring',
              6: 'summer', 7: 'summer', 8: 'summer', 9: 'fall', 10: 'fall', 11: 'fall'}
    d['season'] = d['ts'].dt.month.map(season)
    d['source_disagreement'] = pd.qcut(f['source_sd'].to_numpy(), 3, labels=['low', 'mid', 'high'])
    d['nbm_spread'] = pd.cut(f['nbm_max_sd_f'].to_numpy(), [-1, 2, 3, 99], labels=['<=2F', '3F', '>=4F'])
    d['sky'] = pd.cut(f['nbm_sky_pct'].to_numpy(), [-1, 30, 70, 101], labels=['clear', 'partly', 'cloudy'])
    d['rain_chance'] = pd.cut(f['nbm_precip_prob_12h'].to_numpy(), [-1, 20, 50, 101], labels=['<20%', '20-50%', '>50%'])
    d['afternoon_wind'] = np.where(f['nbm_wind_v_18z'].to_numpy() > 2, 'southerly (sea breeze)',
                                   np.where(f['nbm_wind_v_18z'].to_numpy() < -2, 'northerly', 'light/cross'))
    gap = [abs(means[k] - implied[k][0]) if k in implied and k in means.index else np.nan for k in d['day']]
    d['ml_vs_kalshi_mean_gap'] = pd.cut(np.array(gap), [-1, 1, 2, 99], labels=['<1F', '1-2F', '>2F'])
    rng = np.random.default_rng(config.BOOTSTRAP_SEED)
    table = []
    for factor in ('split', 'season', 'source_disagreement', 'nbm_spread', 'sky', 'rain_chance', 'afternoon_wind',
                   'ml_vs_kalshi_mean_gap'):
        for level, g in d.groupby(factor, observed=True):
            v = g['diff'].to_numpy()
            draws = rng.choice(v, size=(config.BOOTSTRAP_REPLICATES, len(v))).mean(axis=1)
            table.append({'factor': factor, 'level': str(level), 'days': len(v), 'ml_minus_kalshi_brier': v.mean(),
                          'ci_low': np.percentile(draws, 2.5), 'ci_high': np.percentile(draws, 97.5),
                          'ml_better_share': float(np.mean(v < 0))})
    return table


# ---------------------------------------------------------------- trading on disagreement

def trades_for(rows, threshold):
    """Buy YES at the ask when ML exceeds it by threshold; buy NO at 1 - bid when ML is below the bid by threshold."""
    out = []
    for r in rows:
        p, bid, ask, y = float(r['ml_probability']), float(r['market_bid']), float(r['market_ask']), int(r['outcome'])
        if 0 < ask < 1 and p - ask > threshold:
            out.append({'target_date': r['target_date'], 'pnl': y - ask - fee(ask)})
        elif 0 < bid < 1 and bid - p > threshold:
            out.append({'target_date': r['target_date'], 'pnl': (1 - y) - (1 - bid) - fee(1 - bid)})
    return out


def trading_table(rows):
    table = []
    for split in ('train', 'test'):
        subset = [r for r in rows if r['split'] == split]
        days = len({r['target_date'] for r in subset})
        for threshold in THRESHOLDS:
            trades = trades_for(subset, threshold)
            if trades:
                ci = cf.day_bootstrap(trades, lambda ts: float(np.mean([t['pnl'] for t in ts])))
                table.append({'split': split, 'threshold': threshold, 'trades': len(trades), 'days': days,
                              'pnl_per_contract': ci['mean'], 'ci_low': ci['ci_low'], 'ci_high': ci['ci_high'],
                              'pnl_per_day': sum(t['pnl'] for t in trades) / days,
                              'hit_rate': float(np.mean([t['pnl'] > 0 for t in trades]))})
    train = [t for t in table if t['split'] == 'train']
    return table, max(train, key=lambda t: t['pnl_per_contract'] * t['trades'])['threshold']


# ---------------------------------------------------------------- importance

def permutation_importance(frame, repeats=20):
    """Increase in test-day MAE when a group (or feature) is shuffled; model fit on days before the test period."""
    features = ml_model.feature_list()
    train = frame[frame.index <= pd.Timestamp(config.TEST_START) - pd.Timedelta(days=ml_model.GAP_DAYS)]
    test = frame[(frame.index >= pd.Timestamp(config.TEST_START)) & (frame.index <= pd.Timestamp(config.TEST_END))]
    model = ml_model.make_model().fit(train[features], train['residual'])
    mae = lambda x: np.abs(test['high_f'].to_numpy() - test['baseline'].to_numpy() - model.predict(x)).mean()
    base = mae(test[features])
    rng = np.random.default_rng(config.BOOTSTRAP_SEED)

    def shuffled(cols):
        rises = []
        for _ in range(repeats):
            x = test[features].copy()
            x[cols] = x[cols].to_numpy()[rng.permutation(len(x))]
            rises.append(mae(x) - base)
        return float(np.mean(rises)), float(np.std(rises))

    groups = [{'group': 'replace model by consensus', 'mae_increase_f': float(np.abs(test['residual']).mean() - base), 'sd': 0.0}]
    groups += [{'group': g, **dict(zip(('mae_increase_f', 'sd'), shuffled(cols)))} for g, cols in ml_model.GROUPS.items()]
    single = [{'feature': c, **dict(zip(('mae_increase_f', 'sd'), shuffled([c])))} for c in features]
    return base, sorted(groups, key=lambda r: -r['mae_increase_f']), sorted(single, key=lambda r: -r['mae_increase_f'])


# ---------------------------------------------------------------- figures and report

def plot_bars(labels, values, errors, title, xlabel, path, highlight=None):
    fig, ax = plt.subplots(figsize=(7, 0.35 * len(labels) + 1.2))
    colors = ['#c05a2b' if highlight and highlight(v) else '#3b6ea8' for v in values]
    ax.barh(labels[::-1], values[::-1], xerr=None if errors is None else errors[::-1], color=colors[::-1])
    ax.axvline(0, color='#555', lw=0.8)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_monthly(rows, path):
    keys = ['ml', 'noaa_calibrated', 'market']
    frame = pd.DataFrame([{'month': r['target_date'][:7], **{k: (float(r[evaluate.COLUMN[k]]) - int(r['outcome'])) ** 2 for k in keys}}
                          for r in rows]).groupby('month').mean()
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for k, color in zip(keys, ('#c05a2b', '#3b6ea8', '#444444')):
        ax.plot(frame.index, frame[k], marker='o', label=evaluate.LABEL[k], color=color)
    ax.axvline(config.TEST_START.strftime('%Y-%m'), color='#888', ls='--', lw=0.8)
    ax.set_ylabel('Mean Brier per contract')
    ax.set_title('Monthly Brier score (all window days; ML is walk-forward)')
    ax.tick_params(axis='x', rotation=45)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    (OUT / 'figures').mkdir(parents=True, exist_ok=True)
    info = json.loads((OUT / 'model.json').read_text())
    rows = load_rows()
    beta = blend(rows)
    test = [r for r in rows if r['split'] == 'test']
    train = [r for r in rows if r['split'] == 'train']
    result = evaluate.bootstrap(test, MODELS, PAIRS)
    train_result = evaluate.bootstrap(train, MODELS[:-1], PAIRS[:-1])
    write_csv('results/ml/scores.csv', evaluate.results_rows(result, 'test') + evaluate.results_rows(train_result, 'train'))
    write_csv('results/ml/paired_differences.csv', evaluate.pair_rows(result, 'test') + evaluate.pair_rows(train_result, 'train'))

    frame = ml_model.build_frame()
    ml_daily = pd.read_csv(DATA / 'ml_daily.csv')
    temps, implied = temperature_table(rows, frame, ml_daily)
    write_csv('results/ml/temperature_errors.csv', temps)
    write_csv('results/ml/implied_kalshi.csv', [{'target_date': d, 'kalshi_implied_mean_f': m, 'kalshi_implied_sd_f': s}
                                                for d, (m, s) in sorted(implied.items())])
    regime = regimes(rows, frame, implied, ml_daily)
    write_csv('results/ml/regimes.csv', regime)

    records = forecasts.day_records()
    marks = cf.backtest.load_kalshi_marks(records)
    later = {t: marks[rec['day']][i] for rec in records if rec['day'] in marks for i, t in enumerate(rec['tickers'])}
    for r in rows:
        mid = float(r['market_midpoint'])
        r['disagreement'] = float(r['ml_probability']) - mid
        r['surprise'] = int(r['outcome']) - mid
        r['later_move'] = later[r['market_ticker']] - mid if r['market_ticker'] in later else ''
    signal = {s: cf.signal_tests([r for r in rows if r['split'] == s]) for s in ('train', 'test')}
    trading, threshold = trading_table(rows)
    write_csv('results/ml/trading.csv', trading)

    mae, groups, single = permutation_importance(frame)
    write_csv('results/ml/importance_groups.csv', groups)
    write_csv('results/ml/importance_features.csv', single)
    ablation = pd.read_csv(OUT / 'ablation.csv') if (OUT / 'ablation.csv').exists() else None

    figures = OUT / 'figures'
    bins = {m: cf.plot_results.reliability_bins(test, m) for m in ('ml', 'noaa_calibrated', 'market')}
    cf.plot_results.plot_reliability(bins, 'Predicted probability vs realized YES rate (test days)', figures / 'reliability.png', list(bins))
    plot_monthly(rows, figures / 'monthly_brier.png')
    plot_bars([r['group'] for r in groups], [r['mae_increase_f'] for r in groups], [r['sd'] for r in groups],
              'Grouped permutation importance (test days)', 'Increase in MAE when shuffled (°F)', figures / 'importance_groups.png')
    plot_bars([r['feature'] for r in single[:20]], [r['mae_increase_f'] for r in single[:20]], [r['sd'] for r in single[:20]],
              'Top 20 features by permutation importance (test days)', 'Increase in MAE when shuffled (°F)',
              figures / 'importance_features.png')
    if ablation is not None:
        a = ablation[ablation['split'] == 'test'].set_index('run')
        drops = a.drop('all features')
        plot_bars(list(drops.index), list(drops['log_loss'] - a.at['all features', 'log_loss']), None,
                  'Ablation: change in test log loss vs all features', 'Log loss change (positive = worse)',
                  figures / 'ablation.png', highlight=lambda v: v > 0)
    shown = [r for r in regime if r['factor'] != 'split']
    plot_bars([f"{r['factor']}: {r['level']}" for r in shown], [r['ml_minus_kalshi_brier'] for r in shown], None,
              'ML − Kalshi Brier by condition (all window days)', 'Brier difference (negative = ML better)',
              figures / 'regimes.png', highlight=lambda v: v < 0)

    lines = ['# Multi-source ML forecast versus Kalshi (generated by analysis/ml_compare.py; do not edit by hand)', '',
             f"{info['model']}, walk-forward refits every {info['refit_days']} days on {info['history_start']} through D-2, "
             f"{len(info['features'])} features.", '',
             evaluate.markdown(result, 'Accuracy on test days', 'Same contracts for every model. The blend is fit on train days only.'),
             f'Blend weights in log-odds: intercept {beta[0]:.3f}, Kalshi {beta[1]:.3f}, ML {beta[2]:.3f}.', '',
             evaluate.markdown(train_result, 'Accuracy on train days (ML is still out of sample here)',
                               'Walk-forward forecasts, so this is a second, independent period.'),
             '### Point forecast of the high (test days)', '',
             '| Forecast | Days | MAE (°F) | RMSE (°F) | Bias (obs − fcst) | Within 1°F | Within 2°F |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in sorted((t for t in temps if t['split'] == 'test'), key=lambda t: t['mae_f']):
        lines.append(f"| {r['forecast']} | {r['days']} | {r['mae_f']:.2f} | {r['rmse_f']:.2f} | {r['bias_f']:+.2f} | "
                     f"{r['within_1f']:.0%} | {r['within_2f']:.0%} |")
    sel = ml_daily[(ml_daily['model'] == 'ml') & (ml_daily['split'] == 'test')]
    lines += ['', f"Kalshi's noon prices imply a mean sd of {np.mean([s for d, (_, s) in implied.items() if config.split_of(d) == 'test']):.2f}°F; "
              f"the ML mean sd is {sel['sd_f'].mean():.2f}°F and its realized RMSE {math.sqrt((sel['error_f'] ** 2).mean()):.2f}°F.", '',
              '### Feature importance', '',
              f'The model is fit once on days before {config.TEST_START}, then each group is shuffled across test days (20 repeats). '
              f'Test MAE with nothing shuffled is {mae:.2f}°F. Negative means the model\'s use of the group hurts out of sample.', '',
              '| Feature group | MAE increase (°F) |', '| --- | ---: |']
    lines += [f"| {r['group']} | {r['mae_increase_f']:+.3f} ± {r['sd']:.3f} |" for r in groups]
    lines += ['', '| Top single features | MAE increase (°F) |', '| --- | ---: |']
    lines += [f"| {r['feature']} | {r['mae_increase_f']:+.3f} |" for r in single[:10]]
    if ablation is not None:
        tr = ablation[ablation['split'] == 'train'].set_index('run')
        lines += ['', '### Ablation (walk-forward refit without each group)', '',
                  '| Run | Features | Train log loss | Test log loss | Test Brier | Test MAE (°F) |', '| --- | ---: | ---: | ---: | ---: | ---: |']
        lines += [f"| {run} | {r['features']} | {tr.at[run, 'log_loss']:.4f} | {r['log_loss']:.4f} | {r['brier']:.4f} | {r['mae_f']:.2f} |"
                  for run, r in ablation[ablation['split'] == 'test'].set_index('run').iterrows()]
    lines += ['', '### Where ML beats or loses to Kalshi', '',
              'Per-day Brier difference, ML minus the Kalshi noon midpoint, over all window days. Negative means ML is better.', '',
              '| Condition | Level | Days | ML − Kalshi [95% CI] | Days ML better |', '| --- | --- | ---: | --- | ---: |']
    lines += [f"| {r['factor']} | {r['level']} | {r['days']} | {r['ml_minus_kalshi_brier']:+.4f} [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] | "
              f"{r['ml_better_share']:.0%} |" for r in regime]
    lines += ['', '### Does ML − Kalshi disagreement carry signal?', '',
              'Slopes of the outcome surprise, and of the later Kalshi move (noon on the target day), on ML minus the noon midpoint.', '',
              '| Split | Outcome surprise on disagreement | Later Kalshi move on disagreement |', '| --- | --- | --- |']
    lines += [f"| {s} | {cf.cell(signal[s]['outcome_surprise_slope'])} | {cf.cell(signal[s]['later_move_slope'])} |" for s in ('train', 'test')]
    lines += ['', '### Trading Kalshi with the ML forecast', '',
              'Buy YES at the noon ask when ML exceeds it by the threshold, or NO at 1 − bid when ML is below the bid by the threshold. '
              f'Kalshi taker fee {FEE_RATE} × P × (1 − P) per contract included; held to settlement. Threshold chosen on train days: **{threshold}**.', '',
              '| Split | Threshold | Trades | PnL per contract [95% CI] | PnL per day | Hit rate |', '| --- | ---: | ---: | --- | ---: | ---: |']
    lines += [f"| {t['split']} | {t['threshold']:.2f} | {t['trades']} | {t['pnl_per_contract']:+.3f} [{t['ci_low']:+.3f}, {t['ci_high']:+.3f}] | "
              f"{t['pnl_per_day']:+.3f} | {t['hit_rate']:.0%} |" for t in trading]
    text = '\n'.join(lines) + '\n'
    (OUT / 'ml_tables.md').write_text(text, encoding='utf-8')
    (OUT / 'summary.json').write_text(json.dumps({'test': result, 'train': train_result, 'blend_weights': beta.tolist(),
                                                  'signal': signal, 'trading_threshold': threshold}, indent=2))
    print(text)


if __name__ == '__main__':
    main()
