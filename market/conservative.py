"""Train-only sensor-error margin for live running-maximum truncation.

Compares forecast-only, current truncation and conservative truncation on identical
eligible dates and common settings. Default is NYC, whose trade cache can be rebuilt
without querying Kalshi again. Run python -m collect.realtime --cities NYC first.
"""
import argparse
import csv
import datetime as dt
import hashlib
import json
import math
from collections import defaultdict

import numpy as np

import cities
from common import DATA, write_csv
from forecast import probabilities
from market import makers

OUT = DATA / 'results' / 'conservative_truncation'
VARIANTS = ('forecast_only', 'current_truncation', 'conservative_truncation')
MIN_COVERAGE = 0.8
QUANTILE = 0.99  # Fixed before this run; not selected on test PnL.


def reference_markets(code):
    grouped = defaultdict(list)
    with (DATA/'cities'/code/'markets.csv').open(newline='',encoding='utf-8') as f:
        for row in csv.DictReader(f):
            grouped[row['target_date']].append(row)
    output = {}
    for day,rows in grouped.items():
        pairs = [(probabilities.bin_edges(row), row) for row in rows]
        pairs.sort(key=lambda pair: pair[0][0])
        observed = {float(r['settlement_high_f']) for r in rows if r['settlement_high_f'] not in ('','nan')}
        if len(observed)>1:
            raise ValueError(f'{code} {day}: conflicting official settlement temperatures')
        output[day] = {'lows':np.array([p[0][0] for p in pairs]),
                       'highs':np.array([p[0][1] for p in pairs]),
                       'official_high_f':next(iter(observed)) if observed else None}
    return output


def prepare_days(code, days):
    """Use actual +/-infinity contract tails, not finite plotting/median placeholders."""
    reference = reference_markets(code)
    prepared,audit = [],[]
    for original in days:
        info = reference[original['day']]
        day = {**original,'lows':info['lows'],'highs':info['highs']}
        unique_minutes = len(np.unique(np.floor(day['minute_ts']/60)))
        coverage = unique_minutes/1440
        finite_max = day['running_max'][np.isfinite(day['running_max'])]
        sensor_high = float(finite_max.max()) if len(finite_max) else None
        official = info['official_high_f']
        reason = ('insufficient_minute_coverage' if coverage<MIN_COVERAGE else
                  'missing_official_temperature' if official is None else
                  'no_trade_flow' if not day['flow'] else '')
        audit.append({'city':code,'day':day['day'],'split':day['split'],'unique_minutes':unique_minutes,
                      'coverage_fraction':coverage,'sensor_high_f':sensor_high,'official_high_f':official,
                      'sensor_minus_official_f':sensor_high-official if sensor_high is not None and official is not None else None,
                      'eligible':int(not reason),'exclusion_reason':reason})
        if not reason:
            day['official_high_f'] = official
            prepared.append(day)
    return prepared,audit


def latest_fit_date(days):
    first = min(dt.date.fromisoformat(d['day']) for d in days if d['split']=='test')
    # Noon on D-1 cannot know D-1's daily maximum or settlement.
    return (first-dt.timedelta(days=2)).isoformat()


def fit_margin(audit, latest, quantile=QUANTILE):
    if not 0<quantile<=1:
        raise ValueError('Quantile must be in (0, 1]')
    train = [r for r in audit if r['split']=='train' and r['eligible'] and r['day']<=latest]
    if len(train)<20:
        raise ValueError('Need at least 20 eligible, available training days')
    errors = sorted(max(0,float(r['sensor_minus_official_f'])) for r in train)
    value = errors[max(0,math.ceil(quantile*len(errors))-1)]  # Empirical nearest-rank quantile.
    margin = math.ceil(value*2)/2  # Round upward to half degrees, never downward.
    return {'margin_f':margin,'quantile':quantile,'training_days':len(train),
            'first_fit_date':min(r['day'] for r in train),'latest_fit_date':max(r['day'] for r in train),
            'training_fraction_exceeding_margin':sum(e>margin for e in errors)/len(errors),
            'training_max_positive_error_f':max(errors)}


def conservative_tilt(day, ts, margin):
    if margin<0 or not math.isfinite(margin):
        raise ValueError('Margin must be finite and nonnegative')
    weights = np.ones(len(day['highs']))
    index = np.searchsorted(day['minute_ts'],ts-makers.FEED_DELAY,side='right')
    if not index:
        return weights,None
    raw = day['running_max'][index-1]
    safe = raw-margin
    finite = np.isfinite(day['highs'])
    weights[finite & (day['highs']<safe)] = makers.TOUCHED_WEIGHT
    weights[finite & (day['highs']<safe-1)] = makers.DEAD_WEIGHT
    return weights,float(raw)


def choose_common_settings(days, latest):
    train = [d for d in days if d['split']=='train' and d['day']<=latest]
    sd_ref = float(np.median([d['sd'] for d in train]))
    candidates = []
    for b in makers.B_GRID:
        for fee in makers.FEE_GRID:
            pnl = [makers.replay(d,'forecast',b,fee,1.0,sd_ref)['pnl'] for d in train]
            candidates.append({'b0':b,'fee':fee,'train_days':len(pnl),'mean_training_pnl':float(np.mean(pnl))})
    # No tuning of the conservative margin or separate per-maker liquidity settings.
    best = max(candidates,key=lambda r:(r['mean_training_pnl'],-r['b0'],-r['fee']))
    return {**best,'sd_ref':sd_ref},candidates


def run_day(day, variant, settings, margin):
    callback = lambda d,ts: conservative_tilt(d,ts,margin)
    maker = 'forecast' if variant=='forecast_only' else 'forecast_live'
    result = makers.replay(day,maker,settings['b0'],settings['fee'],1.0,settings['sd_ref'],
                           tilt_function=callback if variant=='conservative_truncation' else None)
    weights,_ = ((callback if variant=='conservative_truncation' else makers.tilt)(day,day['flow'][-1][0])
                 if variant!='forecast_only' else (np.ones(len(day['highs'])),None))
    tau = float(weights[day['winner']])
    static_bound = settings['b0']*math.log(1/day['priors']['public'][day['winner']])
    conditional_bound = static_bound+settings['b0']*math.log(1/tau)
    # This conservative bound includes the actual winner's downweight even after sensor error.
    if -result['trading_pnl']>conditional_bound+1e-5:
        raise ValueError('Replay violated the winner-specific tilt loss bound')
    return {'day':day['day'],'split':day['split'],'variant':variant,
            'winner_weight_at_last_quote':tau,
            'winner_heavily_discounted':int(tau<=makers.DEAD_WEIGHT+1e-12),
            'winner_downweighted':int(tau<1), 'winner_specific_bound':conditional_bound,**result}


def bootstrap_draws(days, replicates, block):
    rng = np.random.default_rng(473)
    if block==1:
        return rng.integers(0,len(days),(replicates,len(days)))
    # Resample the FULL calendar, keeping missing-date masks. Restricting starts to
    # fully observed weeks would silently remove isolated-but-valid evaluation days.
    first,last = dt.date.fromisoformat(days[0]),dt.date.fromisoformat(days[-1])
    length = (last-first).days+1
    if length<block:
        raise ValueError('Calendar shorter than the requested block')
    lookup = {day:i for i,day in enumerate(days)}
    calendar = np.array([lookup.get((first+dt.timedelta(days=i)).isoformat(),-1) for i in range(length)])
    starts = rng.integers(0,length-block+1,(replicates,math.ceil(length/block)))
    positions = (starts[:,:,None]+np.arange(block)).reshape(replicates,-1)[:,:length]
    return calendar[positions]


def resampled_means(values, draws):
    values=np.asarray(values,dtype=float)
    sampled=np.where(draws>=0,values[np.maximum(draws,0)],np.nan)
    result=np.nanmean(sampled,axis=1)
    if not np.isfinite(result).all():
        raise ValueError('A bootstrap replicate contains no observed dates')
    return result


def summarize(rows, replicates):
    test = [r for r in rows if r['split']=='test']
    days = sorted({r['day'] for r in test})
    groups = {variant:sorted([r for r in test if r['variant']==variant],key=lambda r:r['day']) for variant in VARIANTS}
    if any([r['day'] for r in group]!=days for group in groups.values()):
        raise ValueError('Makers must use identical test dates')
    summaries,pairs = [],[]
    for block in (1,7):
        draws = bootstrap_draws(days,replicates,block)
        for variant,group in groups.items():
            pnl = np.array([r['pnl'] for r in group])
            means = resampled_means(pnl,draws)
            tail = max(1,math.ceil(.05*len(group)))
            summaries.append({'variant':variant,'bootstrap_block_days':block,'test_days':len(group),
                              'mean_pnl':float(pnl.mean()),'ci_low':float(np.percentile(means,2.5)),
                              'ci_high':float(np.percentile(means,97.5)), 'worst_day':float(pnl.min()),
                              'worst_5pct_mean_pnl':float(np.sort(pnl)[:tail].mean()),'tail_days':tail,
                              'incorrectly_heavily_discounted_winner_days':sum(r['winner_heavily_discounted'] for r in group),
                              'downweighted_winner_days':sum(r['winner_downweighted'] for r in group),
                              **{key:float(np.mean([r[key] for r in group])) for key in ('filled_shares','trading_pnl','fee_income','peak_worst_case_loss')},
                              'pnl_per_share':float(pnl.sum()/sum(r['filled_shares'] for r in group))})
        for first,second in [('current_truncation','forecast_only'),('conservative_truncation','forecast_only'),
                             ('conservative_truncation','current_truncation')]:
            diff = np.array([r['pnl'] for r in groups[first]])-np.array([r['pnl'] for r in groups[second]])
            means = resampled_means(diff,draws)
            pairs.append({'first':first,'second':second,'bootstrap_block_days':block,'test_days':len(days),
                          'mean_pnl_difference':float(diff.mean()),'ci_low':float(np.percentile(means,2.5)),
                          'ci_high':float(np.percentile(means,97.5))})
    return summaries,pairs


def run_city(code, replicates):
    print(f'{code}: preparing observations and replay inputs',flush=True)
    days,audit = prepare_days(code,makers.day_inputs(code))
    latest = latest_fit_date(days)
    margin = fit_margin(audit,latest)
    settings,candidates = choose_common_settings(days,latest)
    print(f'{code}: fixed margin {margin["margin_f"]}F; shared settings {settings}',flush=True)
    rows = [{ 'city':code,**run_day(day,variant,settings,margin['margin_f'])}
            for day in days for variant in VARIANTS]
    summaries,pairs = summarize(rows,replicates)
    prefix = f'results/conservative_truncation/{code}/'
    for name,values in [('sensor_audit',audit),('daily',rows),('summary',summaries),('paired_differences',pairs),('training_settings',candidates)]:
        write_csv(prefix+name+'.csv',values)
    folder = OUT/code
    manifest = {'city':code,'margin':margin,'common_settings':settings,'minimum_minute_coverage':MIN_COVERAGE,
                'last_information_date_allowed_for_fit':latest,'feed_delay_seconds':makers.FEED_DELAY,
                'test_days':sum(d['split']=='test' for d in days),'bootstrap_replicates':replicates,
                'confirmation_status':'exploratory; known NYC failure motivated this experiment',
                'contract_boundary_handling':'True infinite tails restored for every comparison; original plotting placeholders are not knockout boundaries.',
                'input_sha256':{str(p.relative_to(DATA)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                (DATA/'cities'/code/'markets.csv',makers.RAW/code/'asos1min.csv.gz',makers.RAW/code/'trades.csv.gz')}}
    (folder/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    lines = [f'# Conservative truncation: {code}', '',
             '**Exploratory counterfactual replay.** The known test-period failure motivated this extension; '
             'training-only fitting does not turn this into independent confirmation.', '',
             f'Margin: {margin["margin_f"]:g}F from the fixed 99th percentile of positive training sensor errors, '
             f'rounded upward to half degrees ({margin["training_days"]} days through {margin["latest_fit_date"]}). '
             f'Shared b={settings["b0"]}, hypothetical fee={settings["fee"]:.2f}/share. '
             'Both parameter selection and margin fitting stop at D-2 of the first test event.', '',
             'All variants use true open-ended contract bounds and identical coverage-filtered dates. '
             'The conservative rule subtracts the fitted margin from the observed running maximum before '
             'applying the existing soft/hard downweights. It never uses a test settlement high to quote.', '',
             '| Maker | Test days | Mean net PnL/day | Worst day | Worst-5% mean PnL | Winner heavily discounted: days | Shares/day |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in summaries:
        if r['bootstrap_block_days']==1:
            lines.append(f"| {r['variant']} | {r['test_days']} | {r['mean_pnl']:.2f} | {r['worst_day']:.2f} | "
                         f"{r['worst_5pct_mean_pnl']:.2f} | {r['incorrectly_heavily_discounted_winner_days']} | {r['filled_shares']:.0f} |")
    lines += ['', '| Paired PnL difference | Block days | Mean [95% interval] |', '| --- | ---: | --- |']
    for r in pairs:
        lines.append(f"| {r['first']} - {r['second']} | {r['bootstrap_block_days']} | {r['mean_pnl_difference']:.2f} "
                     f"[{r['ci_low']:.2f}, {r['ci_high']:.2f}] |")
    lines += ['', 'Known NYC July 31 failure is retained in daily.csv if it passes the same predeclared '
              'coverage rule as every other date. The margin is not a guaranteed bound on future sensor error. '
              'Winner-specific loss checks include any downweight of the actual winner; no unconditional '
              '1.2b extra-loss claim is made. Fee income is hypothetical and separated in summary.csv.', '']
    (folder/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    print('\n'.join(lines),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cities',nargs='+',default=['NYC'],choices=makers.REALTIME_CITIES)
    parser.add_argument('--replicates',type=int,default=1000)
    args = parser.parse_args()
    if args.replicates<1:
        parser.error('Replicates must be positive')
    for code in args.cities:
        run_city(code,args.replicates)


if __name__=='__main__':
    main()
