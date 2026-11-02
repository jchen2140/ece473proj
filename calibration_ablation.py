"""Controlled GEFS calibration ablation; preserves existing production outputs.

Select one rolling window/spread rule using train-day log loss of the fully corrected
Gaussian. Freeze it across all variants. Verified outcomes are available only through D-2.
The already-examined May–August test period is explicitly an exploratory comparison.
"""
import argparse
import datetime as dt
import hashlib
import json
import math
import statistics
from pathlib import Path

import numpy as np

import calibrate
import config
import evaluate
import forecasts
from collect_nyc import DATA, write_csv

OUT = DATA / 'results' / 'calibration_ablation'
VARIANTS = ('empirical_smoothed', 'gaussian_uncorrected', 'bias_only', 'spread_only', 'bias_and_spread')
LABELS = {'empirical_smoothed': 'Empirical GEFS (common smoothing)',
          'gaussian_uncorrected': 'Uncorrected Gaussian', 'bias_only': 'Bias correction only',
          'spread_only': 'Spread correction only', 'bias_and_spread': 'Bias + spread correction'}
PAIRS = [('gaussian_uncorrected', 'empirical_smoothed'), ('bias_only', 'gaussian_uncorrected'),
         ('spread_only', 'gaussian_uncorrected'), ('bias_and_spread', 'gaussian_uncorrected'),
         ('bias_and_spread', 'bias_only'), ('bias_and_spread', 'spread_only')]


def verified_history(day, window, moments, highs):
    target = dt.date.fromisoformat(day)
    return [(date, moments[date], highs[date])
            for back in range(2, window+2)
            if (date := (target-dt.timedelta(days=back)).isoformat()) in moments and date in highs]


def fits(day, moments, highs, window, spread_method):
    history = verified_history(day, window, moments, highs)
    if len(history) < calibrate.MIN_DAYS:
        return None
    mean, variance = moments[day]
    raw_sd = max(forecasts.MIN_SD, math.sqrt(variance))
    # Estimate centered residual dispersion once, independently of whether the bias
    # shift is applied to the predicted mean. Otherwise spread-only would confound bias.
    corrected = calibrate.predict(day, moments, highs, window, 'bias', spread_method)
    corrected_mean, corrected_sd = corrected
    return {'gaussian_uncorrected': (mean, raw_sd), 'bias_only': (corrected_mean, raw_sd),
            'spread_only': (mean, corrected_sd), 'bias_and_spread': corrected,
            'history': history, 'raw_mean': mean, 'raw_sd': raw_sd}


def validate_probabilities(probabilities, n):
    if len(probabilities) != n or any(not math.isfinite(p) or p <= 0 or p >= 1 for p in probabilities):
        raise ValueError('Invalid opening probabilities')
    if not math.isclose(sum(probabilities), 1, abs_tol=1e-10):
        raise ValueError('Opening probabilities must sum to one')


def predictions(day, group, moments, highs, settings):
    fitted = fits(day, moments, highs, settings['window'], settings['spread_method'])
    if fitted is None:
        return None
    probabilities = {'empirical_smoothed': forecasts.floor_probabilities(
        [float(r['ensemble_probability']) for r in group])}
    for variant in VARIANTS[1:]:
        probabilities[variant] = forecasts.floor_probabilities(forecasts.gaussian_bins(group, *fitted[variant]))
    for p in probabilities.values():
        validate_probabilities(p, len(group))
    return fitted, probabilities


def select_settings(groups, moments, highs):
    # All candidates use exactly the same eligible training days.
    candidates = [(window, spread) for window in calibrate.WINDOWS for spread in calibrate.SPREADS]
    eligible = [d for d in sorted(groups) if config.split_of(d) == 'train'
                and dt.date.fromisoformat(d) >= calibrate.EVAL_START
                and all(fits(d, moments, highs, w, s) is not None for w,s in candidates)]
    if not eligible:
        raise ValueError('No shared training days for selecting calibration settings')
    rows = []
    for window, spread in candidates:
        scores = []
        for day in eligible:
            fitted = fits(day, moments, highs, window, spread)
            ps = forecasts.floor_probabilities(forecasts.gaussian_bins(groups[day], *fitted['bias_and_spread']))
            scores.append(calibrate.losses(ps, [int(r['outcome']) for r in groups[day]]))
        rows.append({'window': window, 'spread_method': spread, 'training_days': len(eligible),
                     'training_start': eligible[0], 'training_end': eligible[-1],
                     'train_brier': statistics.fmean(s[0] for s in scores),
                     'train_log_loss': statistics.fmean(s[1] for s in scores)})
    return min(rows, key=lambda r: (r['train_log_loss'], r['window'], r['spread_method'])), rows


def export_predictions(groups, moments, highs, settings):
    rows, daily, exclusions = [], [], []
    for day, group in sorted(groups.items()):
        result = predictions(day, group, moments, highs, settings)
        if result is None:
            exclusions.append({'target_date': day, 'reason': 'fewer_than_14_verified_history_days'})
            continue
        fitted, probabilities = result
        latest = max(d for d, _, _ in fitted['history'])
        if latest > (dt.date.fromisoformat(day)-dt.timedelta(days=2)).isoformat():
            raise ValueError('Calibration uses information later than D-2')
        winner = next(i for i,r in enumerate(group) if int(r['outcome']) == 1)
        for variant, ps in probabilities.items():
            mean, sd = ((fitted['raw_mean'], fitted['raw_sd']) if variant == 'empirical_smoothed' else fitted[variant])
            losses = [evaluate.contract_losses(p, int(r['outcome'])) for p,r in zip(ps,group)]
            daily.append({'target_date': day, 'split': config.split_of(day), 'forecast_variant': variant,
                          'brier': statistics.fmean(v[0] for v in losses),
                          'log_loss': statistics.fmean(v[1] for v in losses),
                          'winning_bin_probability': ps[winner],
                          'very_low_probability_winner': int(ps[winner] < 0.01),
                          'forecast_mean_f': mean, 'forecast_sd_f': sd})
            for row,p in zip(group,ps):
                rows.append({'target_date': day, 'market_ticker': row['market_ticker'], 'forecast_variant': variant,
                             'split': config.split_of(day), 'probability': p, 'outcome': int(row['outcome']),
                             'prediction_time_utc': row['prediction_time_utc'],
                             'forecast_mean_f': mean, 'forecast_sd_f': sd,
                             'moments_note': 'pre-smoothing empirical moments; sd floored' if variant == 'empirical_smoothed'
                                             else 'pre-uniform-mixture Gaussian parameters',
                             'rolling_window_days': settings['window'], 'spread_method': settings['spread_method'],
                             'calibration_observation_count': len(fitted['history']) if variant in ('bias_only','spread_only','bias_and_spread') else 0,
                             'latest_calibration_observation': latest if variant in ('bias_only','spread_only','bias_and_spread') else '',
                             'uniform_mix': forecasts.UNIFORM_MIX})
    return rows, daily, exclusions


def bootstrap_indices(n, replicates=1000, block_days=1, seed=473):
    if n < 1 or block_days < 1 or block_days > n or replicates < 1:
        raise ValueError('Invalid bootstrap dimensions')
    rng = np.random.default_rng(seed)
    if block_days == 1:
        return rng.integers(0,n,size=(replicates,n))
    # Moving blocks never wrap last day back to the first day of the study.
    starts = rng.integers(0,n-block_days+1,size=(replicates,math.ceil(n/block_days)))
    return (starts[:,:,None]+np.arange(block_days)).reshape(replicates,-1)[:,:n]


def interval(values, draws):
    values = np.asarray(values,dtype=float)
    means = values[draws].mean(axis=1)
    return {'mean': float(values.mean()), 'ci_low': float(np.percentile(means,2.5)),
            'ci_high': float(np.percentile(means,97.5))}


def summarize(daily, replicates):
    test = [r for r in daily if r['split'] == 'test']
    days = sorted({r['target_date'] for r in test})
    if any((dt.date.fromisoformat(b)-dt.date.fromisoformat(a)).days != 1 for a,b in zip(days,days[1:])):
        raise ValueError('Weekly block bootstrap requires consecutive test dates')
    series = {(variant,metric): [next(r[metric] for r in test if r['target_date']==d and r['forecast_variant']==variant)
                                for d in days] for variant in VARIANTS for metric in ('brier','log_loss')}
    summaries, paired = [], []
    for block in (1,7):
        draws = bootstrap_indices(len(days),replicates,block,config.BOOTSTRAP_SEED)
        for variant in VARIANTS:
            stats = {metric: interval(series[variant,metric],draws) for metric in ('brier','log_loss')}
            summaries.append({'forecast_variant': variant, 'bootstrap_block_days': block, 'test_days': len(days),
                              'test_contracts': len(days)*6,
                              **{f'{metric}_{field}': value for metric,s in stats.items() for field,value in s.items()},
                              'very_low_probability_winner_days': sum(r['very_low_probability_winner'] for r in test if r['forecast_variant']==variant)})
        for a,b in PAIRS:
            for metric in ('brier','log_loss'):
                diff = np.asarray(series[a,metric])-series[b,metric]
                paired.append({'first': a, 'second': b, 'metric': metric, 'bootstrap_block_days': block,
                               'test_days': len(days), **interval(diff,draws)})
    return summaries, paired


def write_report(summaries, paired, settings):
    lines = ['# Controlled calibration experiment', '',
             '**Exploratory results:** this test period had already been examined before designing the ablation. '
             'Confirmation on another period or city is required.', '',
             f"Shared rolling window: {settings['window']} days; spread rule: {settings['spread_method']}. "
             'Chosen once on training days using fully corrected Gaussian log loss, then frozen for all variants. '
             'Rolling coefficients update using verified outcomes through D-2, including earlier test-period outcomes as time advances.', '',
             f'All variants use the same {forecasts.UNIFORM_MIX:.0%} uniform mixture and the same eligible test days. '
             'The four Gaussian variants form a 2×2 mean/spread experiment; empirical versus uncorrected Gaussian '
             'isolates the distribution-form change. Gaussian SD has a common 1.5°F floor.', '',
             '| Variant | Test days | Brier | Binary log loss | Days with winning-bin probability <1% |',
             '| --- | ---: | ---: | ---: | ---: |']
    for r in summaries:
        if r['bootstrap_block_days']==1:
            lines.append(f"| {LABELS[r['forecast_variant']]} | {r['test_days']} | {r['brier_mean']:.4f} | "
                         f"{r['log_loss_mean']:.4f} | {r['very_low_probability_winner_days']} |")
    lines += ['', '## Paired uncertainty', '',
              'Negative differences favor the first variant. Weekly moving-block intervals assess sensitivity '
              'to dependence across consecutive weather days; six bins of each day are kept together.', '',
              '| First − second | Metric | Block length | Difference [95% interval] |',
              '| --- | --- | ---: | --- |']
    for r in paired:
        lines.append(f"| {r['first']} − {r['second']} | {r['metric']} | {r['bootstrap_block_days']} | "
                     f"{r['mean']:.4f} [{r['ci_low']:.4f}, {r['ci_high']:.4f}] |")
    lines += ['', 'These forecast variants are exported in `probabilities.csv` with model parameters and '
              'calibration-observation provenance. Forecast parameters describe the underlying distribution '
              'before the common uniform mixture. The 1% threshold is a descriptive diagnostic, not a trading rule.', '']
    (OUT/'report.md').write_text('\n'.join(lines),encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replicates',type=int,default=config.BOOTSTRAP_REPLICATES)
    args = parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    groups, moments = forecasts.day_groups(), forecasts.ensemble_moments()
    highs = {d:float(g[0]['kalshi_settlement_high_f']) for d,g in groups.items()}
    settings, candidates = select_settings(groups,moments,highs)
    rows,daily,exclusions = export_predictions(groups,moments,highs,settings)
    summaries,paired = summarize(daily,args.replicates)
    prefix = 'results/calibration_ablation/'
    for name,values in [('probabilities',rows),('daily_scores',daily),('candidate_settings',candidates),
                        ('forecast_scores',summaries),('paired_differences',paired),('excluded_days',exclusions)]:
        if values:
            write_csv(prefix+name+'.csv',values)
    source_names = ('matched_with_climatology.csv','ensemble_daily_maxima.csv')
    manifest = {'selected_on_train_only':settings,'variants':VARIANTS,'uniform_mix':forecasts.UNIFORM_MIX,
                'minimum_gaussian_sd_f':forecasts.MIN_SD, 'bootstrap_replicates':args.replicates,
                'forecast_days':len(daily)//len(VARIANTS),'exported_rows':len(rows),
                'test_days':len({r['target_date'] for r in daily if r['split']=='test'}),
                'confirmation_status':'exploratory_previously_examined_test_period',
                'hyperparameter_selection':'Fully corrected Gaussian train log loss; shared by every variant, not independently optimized per variant.',
                'input_sha256':{n:hashlib.sha256((DATA/n).read_bytes()).hexdigest() for n in source_names}}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    write_report(summaries,paired,settings)
    print(json.dumps(manifest,indent=2))
    print('Saved controlled forecast experiment to',OUT)


if __name__ == '__main__':
    main()
