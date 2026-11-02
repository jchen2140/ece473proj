"""Replay controlled forecast variants with common fees and equal opening loss budgets.

Uses the existing counterfactual taker-trade replay, without changing its fill assumptions.
The main fee and fixed-b reference reuse the previously train-selected NOAA settings.
The $300 risk budget is an illustrative fixed experimental control, not an optimized policy.
"""
import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict

import numpy as np

import backtest
import config
import forecasts
import mm_metrics
from calibration_ablation import OUT, VARIANTS, LABELS, PAIRS, bootstrap_indices, interval, validate_probabilities
from collect_nyc import DATA, write_csv

MODELS = VARIANTS + ('kalshi',)


def liquidity_for_budget(probabilities, budget):
    validate_probabilities(probabilities,len(probabilities))
    if not math.isfinite(budget) or budget <= 0:
        raise ValueError('Loss budget must be positive and finite')
    return budget / math.log(1/min(probabilities))


def records_from_export(path=OUT/'probabilities.csv'):
    with path.open(newline='',encoding='utf-8') as f:
        exported = list(csv.DictReader(f))
    probability = {}
    for r in exported:
        key = (r['forecast_variant'], r['market_ticker'])
        if key in probability:
            raise ValueError('Duplicate variant/ticker in probability export')
        probability[key] = float(r['probability'])
    records, excluded = [], []
    for day,group in forecasts.day_groups().items():
        if config.split_of(day) != 'test':
            continue
        if any(r['market_midpoint']=='' for r in group):
            excluded.append({'day':day,'reason':'missing_market_quote'})
            continue
        if any((v,r['market_ticker']) not in probability for v in VARIANTS for r in group):
            excluded.append({'day':day,'reason':'missing_variant_probability'})
            continue
        seeds = {v:[probability[v,r['market_ticker']] for r in group] for v in VARIANTS}
        seeds['kalshi'] = forecasts.floor_probabilities([float(r['market_midpoint']) for r in group])
        for ps in seeds.values():
            validate_probabilities(ps,len(group))
        records.append({'day':day,'tickers':[r['market_ticker'] for r in group],
                        'winner':next(i for i,r in enumerate(group) if r['outcome']=='1'),
                        'cutoff':group[0]['prediction_time_utc'],'seeds':seeds})
    return records,excluded


def risk_summary(values):
    pnl = np.asarray([r['pnl'] for r in values])
    count = max(1,math.ceil(len(pnl)*0.05))
    return {'mean_daily_pnl':float(pnl.mean()),
            'worst_5pct_mean_pnl':float(np.sort(pnl)[:count].mean()),
            'worst_5pct_days':count,'worst_day_pnl':float(pnl.min()),
            **{key:float(np.mean([r[key] for r in values]))
               for key in ('trading_pnl','fee_income','filled_shares','opening_brier','closing_brier',
                           'opening_loss_bound','b','peak_worst_case_loss')}}


def run(records,flow,budget,fixed_b,fees):
    rows = []
    for mode in ('equal_loss_budget','fixed_b'):
        for fee in fees:
            print(f'Replay: {mode}, fee={fee:g}',flush=True)
            for record in records:
                offered = flow[record['day']]
                for variant in MODELS:
                    ps = record['seeds'][variant]
                    b = liquidity_for_budget(ps,budget) if mode=='equal_loss_budget' else fixed_b
                    bound = b*math.log(1/min(ps))
                    state = backtest.replay_day(offered,ps,b,fee,backtest.stop_time(record['day'],backtest.MAIN_STOP))
                    metrics = mm_metrics.day_metrics(state,record['winner'],ps)
                    if metrics['trading_pnl'] < -bound-1e-5:
                        raise ValueError('Replay exceeded theoretical opening loss bound')
                    rows.append({'target_date':record['day'],'forecast_variant':variant,'risk_mode':mode,
                                 'fee':fee,'b':b,'opening_loss_bound':bound,
                                 'winning_bin_probability':ps[record['winner']],
                                 'very_low_probability_winner':int(ps[record['winner']]<0.01),**metrics})
    return rows


def summaries_and_pairs(rows,replicates):
    groups = defaultdict(list)
    for row in rows:
        groups[row['risk_mode'],row['fee'],row['forecast_variant']].append(row)
    summaries,paired,diagnostics = [],[],[]
    days = sorted({r['target_date'] for r in rows})
    if any((config.as_date(b)-config.as_date(a)).days!=1 for a,b in zip(days,days[1:])):
        raise ValueError('Weekly block sensitivity requires consecutive dates')
    for (mode,fee,variant),values in sorted(groups.items()):
        values.sort(key=lambda r:r['target_date'])
        if [r['target_date'] for r in values] != days:
            raise ValueError('Replay comparisons must use identical days')
        stats = risk_summary(values)
        for block in (1,7):
            draws = bootstrap_indices(len(days),replicates,block,config.BOOTSTRAP_SEED)
            pnl = interval([r['pnl'] for r in values],draws)
            summaries.append({'risk_mode':mode,'fee':fee,'forecast_variant':variant,
                              'bootstrap_block_days':block,'test_days':len(days),**stats,
                              'pnl_ci_low':pnl['ci_low'],'pnl_ci_high':pnl['ci_high']})
        # Conditioned on the realized winner: descriptive mechanism check, not an ex-ante trading rule.
        for low in (False,True):
            subset = [r for r in values if bool(r['very_low_probability_winner'])==low]
            diagnostics.append({'risk_mode':mode,'fee':fee,'forecast_variant':variant,
                                'realized_winner_probability_group':'below_1pct' if low else 'at_least_1pct',
                                'days':len(subset),
                                'mean_daily_pnl':float(np.mean([r['pnl'] for r in subset])) if subset else None,
                                'mean_trading_pnl':float(np.mean([r['trading_pnl'] for r in subset])) if subset else None})
    for mode,fee in sorted({(r['risk_mode'],r['fee']) for r in rows}):
        pairs = PAIRS + [('bias_and_spread','kalshi')]
        for block in (1,7):
            draws = bootstrap_indices(len(days),replicates,block,config.BOOTSTRAP_SEED)
            for a,b in pairs:
                diff = np.array([r['pnl'] for r in groups[mode,fee,a]])-np.array([r['pnl'] for r in groups[mode,fee,b]])
                paired.append({'risk_mode':mode,'fee':fee,'first':a,'second':b,
                               'bootstrap_block_days':block,'test_days':len(days),**interval(diff,draws)})
    return summaries,paired,diagnostics


def report(summaries,main_fee,budget,fixed_b):
    labels = {**LABELS,'kalshi':'Kalshi midpoint (common smoothing)'}
    lines = ['# Calibration and market-making experiment', '',
             '**Exploratory counterfactual replay, not realized trading performance.** '
             'The current test period was already inspected before this experiment was designed.', '',
             f'Main fee: ${main_fee:.2f}/share, reused from the existing train-selected NOAA replay. '
             f'Equal opening loss budget: ${budget:g}/event. Fixed-b reference: {fixed_b:g}. '
             'No settings were optimized against these new test results. Fee-free runs are separate replays '
             'with different fills, not subtraction of fees from a fee-charging run.', '',
             'The fee is a hypothetical charge collected by this maker, not an assertion about Kalshi exchange fees. '
             'Quotes halt at noon on the target day. All variants face the same observed arrivals; '
             'traders may behave differently in reality.', '']
    for mode in ('equal_loss_budget','fixed_b'):
        lines += [f'## {mode}', '',
                  '| Variant | Opening Brier | Trading PnL/day | Fee income/day | Net PnL/day [95% day interval] | Worst-5% mean PnL | Shares/day | Mean loss bound |',
                  '| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |']
        for r in summaries:
            if r['risk_mode']==mode and r['fee']==main_fee and r['bootstrap_block_days']==1:
                lines.append(f"| {labels[r['forecast_variant']]} | {r['opening_brier']:.4f} | ${r['trading_pnl']:.2f} | "
                             f"${r['fee_income']:.2f} | ${r['mean_daily_pnl']:.2f} [${r['pnl_ci_low']:.2f}, ${r['pnl_ci_high']:.2f}] | "
                             f"${r['worst_5pct_mean_pnl']:.2f} | {r['filled_shares']:.0f} | ${r['opening_loss_bound']:.2f} |")
    lines += ['', 'CSV outputs include zero-fee sensitivity, paired PnL differences, and seven-day moving-block '
              'intervals. Positive paired PnL differences favor the first variant. Worst-5% PnL averages the '
              'six worst days out of 105; a less negative value is better. This small tail sample is noisy.', '',
              'Equal loss budgets control the worst-case opening bound, not equal turnover or equal realized risk. '
              'Within fixed-b runs only b and fee are shared; their loss bounds differ by forecast sharpness.', '',
              '`winner_probability_diagnostics.csv` conditions on the **realized** winning bin and is descriptive '
              'only; its groups cannot be used as an available-at-cutoff trading signal.', '']
    (OUT/'replay_report.md').write_text('\n'.join(lines),encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--budget',type=float,default=300.0)
    parser.add_argument('--replicates',type=int,default=config.BOOTSTRAP_REPLICATES)
    args = parser.parse_args()
    if not math.isfinite(args.budget) or args.budget<=0:
        parser.error('Budget must be positive and finite')
    with (DATA/'results/backtest/summary.json').open() as f:
        reference = json.load(f)['selected']['noaa_calibrated']
    fees = sorted({0.0,float(reference['fee'])})
    records,excluded = records_from_export()
    if not records:
        raise ValueError('No complete test records')
    print(f'Loading trade flow for {len(records)} identical test days',flush=True)
    flow = backtest.load_flow(records)
    missing = [r['day'] for r in records if not flow.get(r['day'])]
    if missing:
        raise ValueError(f'Incomplete trade flow: {missing}')
    rows = run(records,flow,args.budget,float(reference['b']),fees)
    summaries,paired,diagnostics = summaries_and_pairs(rows,args.replicates)
    prefix = 'results/calibration_ablation/'
    for name,values in [('replay_daily',rows),('replay_summary',summaries),('replay_paired_differences',paired),
                        ('winner_probability_diagnostics',diagnostics),('replay_excluded_days',excluded)]:
        if values:
            write_csv(prefix+name+'.csv',values)
    manifest = {'test_days':len(records),'excluded_days':excluded,'models':MODELS,'equal_loss_budget':args.budget,
                'fixed_b_reference':reference['b'],'fees':fees,'fee_source':'Existing train-selected calibrated NOAA replay; shared across every variant.',
                'budget_selection':'Illustrative fixed budget, not optimized on test outcomes.',
                'risk_modes':['equal_loss_budget','fixed_b'],'bootstrap_replicates':args.replicates,
                'bootstrap_block_days':[1,7],'confirmation_status':'exploratory_previously_examined_test_period',
                'input_sha256':{str(path.relative_to(DATA)):hashlib.sha256(path.read_bytes()).hexdigest()
                                for path in (OUT/'probabilities.csv',DATA/'flow/trades.csv.gz',DATA/'results/backtest/summary.json')}}
    (OUT/'replay_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    report(summaries,float(reference['fee']),args.budget,float(reference['b']))
    print('Saved controlled replay to',OUT,flush=True)


if __name__ == '__main__':
    main()
