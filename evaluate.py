"""Out-of-sample evaluation: test days only, every model scored on the same contract rows.

Reads data/matched_with_climatology.csv (from milestone1.py) and data/historical_tmax.csv.
Scores the raw ensemble, climatology, persistence and the market midpoint. Intervals come
from resampling whole test days (the six contracts of one day are not independent).
Writes tables to data/results/.
"""
import csv
import datetime as dt
import json
import math
from collections import defaultdict

import numpy as np

import config
from collect_nyc import DATA, write_csv
from climatology import load_history
from milestone1 import strike
from persistence import Persistence

RESULTS = DATA / 'results'
EPS = 1e-6  # Same log-loss clip as milestone1.score.

MODELS = [('raw_ensemble', 'GEFS ensemble'), ('climatology', 'Climatology'),
          ('persistence', 'Persistence'), ('market', 'Kalshi midpoint')]
LABEL = dict(MODELS)
COLUMN = {'raw_ensemble': 'ensemble_probability', 'climatology': 'climatology_probability',
          'persistence': 'persistence_probability', 'market': 'market_midpoint'}
# Each pair is reported as (first − second); negative means the first model scores better.
PAIRS = [('raw_ensemble', 'climatology'), ('raw_ensemble', 'market'), ('raw_ensemble', 'persistence')]


def read_csv(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def contract_losses(p, y, eps=EPS):
    q = min(1 - eps, max(eps, p))
    return (p - y) ** 2, -(y * math.log(q) + (1 - y) * math.log(1 - q))


def day_totals(rows, models, eps=EPS):
    """Per-day sums of Brier and log loss for each model, plus contract counts."""
    days = sorted({r['target_date'] for r in rows})
    index = {d: i for i, d in enumerate(days)}
    counts = np.zeros(len(days))
    sums = {m: np.zeros((2, len(days))) for m in models}
    for r in rows:
        i = index[r['target_date']]
        counts[i] += 1
        for m in models:
            b, l = contract_losses(float(r[COLUMN[m]]), int(r['outcome']), eps)
            sums[m][0, i] += b
            sums[m][1, i] += l
    return days, counts, sums


def bootstrap(rows, models, pairs, replicates=config.BOOTSTRAP_REPLICATES, seed=config.BOOTSTRAP_SEED, eps=EPS):
    """Resample days with replacement; return point estimates and 95% percentile intervals."""
    days, counts, sums = day_totals(rows, models, eps)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(days), size=(replicates, len(days)))
    n = counts[draws].sum(axis=1)
    point, reps = {}, {}
    for m in models:
        point[m] = sums[m].sum(axis=1) / counts.sum()
        reps[m] = np.stack([sums[m][k][draws].sum(axis=1) / n for k in range(2)])
    out = {'days': len(days), 'contracts': int(counts.sum()), 'replicates': replicates, 'models': {}, 'pairs': {}}
    for m in models:
        out['models'][m] = {metric: {'mean': float(point[m][k]),
                                     'ci_low': float(np.percentile(reps[m][k], 2.5)),
                                     'ci_high': float(np.percentile(reps[m][k], 97.5))}
                            for k, metric in enumerate(('brier', 'log_loss'))}
    for a, b in pairs:
        if a in models and b in models:
            out['pairs'][f'{a}-{b}'] = {metric: {'mean': float(point[a][k] - point[b][k]),
                                                 'ci_low': float(np.percentile(reps[a][k] - reps[b][k], 2.5)),
                                                 'ci_high': float(np.percentile(reps[a][k] - reps[b][k], 97.5)),
                                                 'share_of_replicates_first_better': float(np.mean(reps[a][k] < reps[b][k]))}
                                        for k, metric in enumerate(('brier', 'log_loss'))}
    return out


def fmt(cell):
    return f"{cell['mean']:.4f} [{cell['ci_low']:.4f}, {cell['ci_high']:.4f}]"


def results_rows(result, comparison):
    rows = []
    for m, s in result['models'].items():
        rows.append({'comparison': comparison, 'model': m, 'label': LABEL[m], 'days': result['days'],
                     'contracts': result['contracts'],
                     'brier': s['brier']['mean'], 'brier_ci_low': s['brier']['ci_low'], 'brier_ci_high': s['brier']['ci_high'],
                     'log_loss': s['log_loss']['mean'], 'log_loss_ci_low': s['log_loss']['ci_low'],
                     'log_loss_ci_high': s['log_loss']['ci_high']})
    return rows


def pair_rows(result, comparison):
    rows = []
    for key, s in result['pairs'].items():
        a, b = key.split('-')
        rows.append({'comparison': comparison, 'difference': f'{a} − {b}', 'days': result['days'],
                     'contracts': result['contracts'],
                     'brier_diff': s['brier']['mean'], 'brier_ci_low': s['brier']['ci_low'],
                     'brier_ci_high': s['brier']['ci_high'],
                     'brier_share_first_better': s['brier']['share_of_replicates_first_better'],
                     'log_loss_diff': s['log_loss']['mean'], 'log_loss_ci_low': s['log_loss']['ci_low'],
                     'log_loss_ci_high': s['log_loss']['ci_high'],
                     'log_loss_share_first_better': s['log_loss']['share_of_replicates_first_better']})
    return rows


def markdown(result, title, note):
    lines = [f'### {title}', '', note, '',
             f"{result['days']} days, {result['contracts']} contracts, {result['replicates']} day-level bootstrap replicates. "
             'Lower is better; brackets are 95% intervals.', '',
             '| Model | Days | Contracts | Brier [95% CI] | Log loss [95% CI] |', '| --- | ---: | ---: | --- | --- |']
    for m, s in result['models'].items():
        lines.append(f"| {LABEL[m]} | {result['days']} | {result['contracts']} | {fmt(s['brier'])} | {fmt(s['log_loss'])} |")
    if result['pairs']:
        lines += ['', '| Difference (first − second) | Brier diff [95% CI] | Log loss diff [95% CI] | '
                  'Share of replicates where first is better (Brier) |', '| --- | --- | --- | ---: |']
        for key, s in result['pairs'].items():
            a, b = key.split('-')
            lines.append(f"| {LABEL[a]} − {LABEL[b]} | {fmt(s['brier'])} | {fmt(s['log_loss'])} | "
                         f"{s['brier']['share_of_replicates_first_better']:.3f} |")
    return '\n'.join(lines) + '\n'


def main():
    rows = read_csv(DATA / 'matched_with_climatology.csv')
    persistence = Persistence(load_history())
    for r in rows:
        day = dt.date.fromisoformat(r['target_date'])
        p = persistence.probability(day, strike(r))
        r['persistence_probability'] = '' if p is None else p
        r['split'] = config.split_of(day) or ''
    for day in {r['target_date'] for r in rows}:
        values = [r['persistence_probability'] for r in rows if r['target_date'] == day]
        if '' not in values and not math.isclose(sum(values), 1, abs_tol=1e-9):
            raise ValueError(f'{day}: persistence probabilities do not sum to one')
    RESULTS.mkdir(parents=True, exist_ok=True)
    write_csv('results/scored_rows.csv', rows)

    models = [m for m, _ in MODELS]
    has = lambda r, ms: all(r[COLUMN[m]] != '' for m in ms)
    test_rows = [r for r in rows if r['split'] == 'test']
    shared = [r for r in test_rows if has(r, models)]
    missing = {m: sum(r[COLUMN[m]] == '' for r in test_rows) for m in models}

    # Headline: test days, identical rows (a contract is dropped for every model if any model lacks it).
    headline = bootstrap(shared, models, PAIRS)
    # Supplement 1: no market quote required, so more test contracts.
    no_market_models = [m for m in models if m != 'market']
    no_market = bootstrap([r for r in test_rows if has(r, no_market_models)], no_market_models, PAIRS)
    # Supplement 2: whole window. None of these models has parameters fit on 2025-26 dates.
    window = bootstrap([r for r in rows if r['split'] and has(r, models)], models, PAIRS)

    # Log loss depends on the clip when a forecast gives exactly 0 to the bin that wins.
    # The raw ensemble does this whenever all 31 members miss; show how much that matters.
    zero_misses = [r for r in shared if float(r['ensemble_probability']) == 0 and r['outcome'] == '1']
    clip = {eps: bootstrap(shared, models, PAIRS, eps=eps) for eps in (1e-3, 1e-2)}

    write_csv('results/results_table.csv', results_rows(headline, 'test_shared_rows')
              + results_rows(no_market, 'test_quote_not_required') + results_rows(window, 'whole_window'))
    write_csv('results/paired_differences.csv', pair_rows(headline, 'test_shared_rows')
              + pair_rows(no_market, 'test_quote_not_required') + pair_rows(window, 'whole_window'))

    md = ['# Results tables (generated by evaluate.py; do not edit by hand)', '',
          f'Train {config.TRAIN_START} to {config.TRAIN_END}; test {config.TEST_START} to {config.TEST_END}. '
          'Mean Brier score and binary log loss per contract (log loss clips probabilities to [1e-6, 1-1e-6]). '
          f'Intervals resample whole days with replacement ({config.BOOTSTRAP_REPLICATES} replicates, seed {config.BOOTSTRAP_SEED}).', '',
          markdown(headline, 'Main comparison: test days, identical rows',
                   'Every model is scored on the same contracts: test-day contracts with a Kalshi midpoint at the cutoff '
                   'and a persistence forecast.'),
          '### Log-loss sensitivity to the probability clip (test days, identical rows)', '',
          f'{len(zero_misses)} of {len(shared)} test contracts resolved YES while the raw ensemble gave them probability 0 '
          '(all 31 members outside the bin). Each costs -ln(clip) in log loss, so the ensemble log loss depends on the clip. '
          'Brier scores do not change.', '',
          '| Clip | GEFS ensemble | Climatology | Kalshi midpoint | Ensemble − Climatology [95% CI] |',
          '| --- | --- | --- | --- | --- |',
          *[f"| {eps:g} | {r['models']['raw_ensemble']['log_loss']['mean']:.4f} | {r['models']['climatology']['log_loss']['mean']:.4f} | "
            f"{r['models']['market']['log_loss']['mean']:.4f} | {fmt(r['pairs']['raw_ensemble-climatology']['log_loss'])} |"
            for eps, r in [(EPS, headline), *clip.items()]], '',
          markdown(no_market, 'Supplement: test days without requiring a market quote',
                   'All test contracts that the three non-market models can score; the market is left out.'),
          markdown(window, 'Supplement: whole window (train and test days)',
                   'The ensemble is uncorrected, climatology uses 1991–2020 and persistence uses 1991–2020 changes, so none '
                   'of these is fit on 2025–26 data and every day is out of sample for them. Kept separate from the '
                   'main table because the train days will be used for fitting later.')]
    (RESULTS / 'results_tables.md').write_text('\n'.join(md))

    summary = {'train_period': [config.TRAIN_START.isoformat(), config.TRAIN_END.isoformat()],
               'test_period': [config.TEST_START.isoformat(), config.TEST_END.isoformat()],
               'collected_days': {'train': len({r['target_date'] for r in rows if r['split'] == 'train'}),
                                  'test': len({r['target_date'] for r in test_rows})},
               'test_contracts_total': len(test_rows), 'test_contracts_shared': len(shared),
               'test_contracts_missing_by_model': missing,
               'test_zero_probability_yes_contracts': len(zero_misses),
               'headline': headline, 'supplement_no_market': no_market, 'supplement_whole_window': window,
               'log_loss_clip_sensitivity': {str(eps): r for eps, r in clip.items()}}
    (RESULTS / 'summary.json').write_text(json.dumps(summary, indent=2))
    print((RESULTS / 'results_tables.md').read_text())
    print(json.dumps({k: summary[k] for k in ('collected_days', 'test_contracts_total', 'test_contracts_shared',
                                              'test_contracts_missing_by_model')}, indent=2))


if __name__ == '__main__':
    main()
