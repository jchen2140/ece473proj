"""Frozen-rule CHI replication of the conservative truncation experiment.

Reads the saved pre-run protocol and refuses to run changed strategy code. Adds paired
tail-mean uncertainty without modifying the prediction or trading rules.
"""
import csv
import datetime as dt
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from common import ROOT,write_csv
from market import conservative

OUT=conservative.OUT
PROTOCOL=OUT/'cross_city_protocol.json'


def verify_protocol(path=PROTOCOL):
    protocol=json.loads(path.read_text())
    for name,expected in protocol['code_sha256'].items():
        actual=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
        if actual!=expected:
            raise ValueError(f'Frozen strategy code changed: {name}; do not silently retune confirmation')
    if protocol['replication_city']!='CHI' or protocol['coverage_fraction']!=conservative.MIN_COVERAGE or conservative.QUANTILE!=.99:
        raise ValueError('Protocol and implementation disagree')
    return protocol


def read_city(code,name):
    with (OUT/code/name).open(newline='',encoding='utf-8') as f:
        return list(csv.DictReader(f))


def tail_mean(values):
    values=np.asarray(values,dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError('Tail mean requires nonempty finite observations')
    return float(np.sort(values)[:max(1,math.ceil(.05*len(values)))].mean())


def paired_tail_intervals(first,second,draws):
    """Resample the same dates; each maker's own worst tail is recomputed in each draw."""
    a,b=np.asarray(first,dtype=float),np.asarray(second,dtype=float)
    if len(a)!=len(b):
        raise ValueError('Paired samples must have equal lengths')
    differences=[]
    for indexes in draws:
        indexes=indexes[indexes>=0]
        differences.append(tail_mean(a[indexes])-tail_mean(b[indexes]))
    return {'mean_difference':tail_mean(a)-tail_mean(b),
            'ci_low':float(np.percentile(differences,2.5)),
            'ci_high':float(np.percentile(differences,97.5))}


def compare_city(code,replicates=1000):
    daily=read_city(code,'daily.csv')
    groups={v:sorted([r for r in daily if r['split']=='test' and r['variant']==v],key=lambda r:r['day'])
            for v in conservative.VARIANTS}
    days=[r['day'] for r in groups['forecast_only']]
    if any([r['day'] for r in rows]!=days for rows in groups.values()):
        raise ValueError('Makers do not use identical date sets')
    paired=[]
    for block in (1,7):
        draws=conservative.bootstrap_draws(days,replicates,block)
        for other in ('current_truncation','forecast_only'):
            a=np.array([float(r['pnl']) for r in groups['conservative_truncation']])
            b=np.array([float(r['pnl']) for r in groups[other]])
            mean_draws=conservative.resampled_means(a-b,draws)
            paired.append({'city':code,'first':'conservative_truncation','second':other,
                           'metric':'mean_net_pnl','bootstrap_block_days':block,'test_days':len(days),
                           'mean_difference':float(np.mean(a-b)),
                           'ci_low':float(np.percentile(mean_draws,2.5)),
                           'ci_high':float(np.percentile(mean_draws,97.5))})
            paired.append({'city':code,'first':'conservative_truncation','second':other,
                           'metric':'worst_5pct_mean_pnl','bootstrap_block_days':block,'test_days':len(days),
                           **paired_tail_intervals(a,b,draws)})
    summary=[]
    manifest=json.loads((OUT/code/'manifest.json').read_text())
    for variant,rows in groups.items():
        pnl=[float(r['pnl']) for r in rows]
        volume=sum(float(r['filled_shares']) for r in rows)
        summary.append({'city':code,'margin_f':manifest['margin']['margin_f'],
                        'fitting_days':manifest['margin']['training_days'],
                        'b0':manifest['common_settings']['b0'],'fee':manifest['common_settings']['fee'],
                        'variant':variant,'test_days':len(rows),'mean_pnl':float(np.mean(pnl)),
                        'worst_day':min(pnl),'worst_5pct_mean_pnl':tail_mean(pnl),
                        'tail_days':math.ceil(.05*len(rows)),
                        'incorrectly_heavily_discounted_winner_days':sum(int(r['winner_heavily_discounted']) for r in rows),
                        'mean_filled_shares':volume/len(rows),'net_pnl_per_share':sum(pnl)/volume,
                        'mean_trading_pnl':float(np.mean([float(r['trading_pnl']) for r in rows])),
                        'mean_fee_income':float(np.mean([float(r['fee_income']) for r in rows]))})
    return summary,paired


def write_report(protocol,summaries,paired):
    lines=['# Frozen-rule conservative-truncation replication', '',
           'CHI (Chicago Midway) was selected before inspecting its conservative replay results. '
           'The 99% training-error quantile, upward half-degree rounding, 80% minute coverage and five-minute '
           'delay were frozen. Each city fits its own numerical margin and common liquidity/fee setting '
           'using available training data only. No strategy change was made after Chicago results.', '',
           '**Scope:** out-of-city replication of a frozen procedure, not a fully blind external validation. '
           'The team had previously examined unguarded Chicago aggregate results. The NYC failure motivated the design.', '',
           'The margin column is the city-fitted value; only conservative truncation applies it. '
           'Winner counts refer to the heavy 0.001 downweight, not the softer 0.3 discount.', '',
           '| City | Maker | Margin | Test days | Mean net PnL/day | Worst-5% mean PnL | Worst day | Heavily discounted winners | Shares/day |',
           '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in summaries:
        lines.append(f"| {r['city']} | {r['variant']} | {r['margin_f']:g}F | {r['test_days']} | {r['mean_pnl']:.2f} | "
                     f"{r['worst_5pct_mean_pnl']:.2f} | {r['worst_day']:.2f} | "
                     f"{r['incorrectly_heavily_discounted_winner_days']} | {r['mean_filled_shares']:.0f} |")
    lines+=['', '## Paired weekly-block uncertainty', '',
            'Positive differences favor conservative truncation. Each tail statistic is recomputed inside '
            'each shared-date resample. These are percentile intervals conditional on fitted strategies; '
            'very small tail samples limit precision. Missing-date masks are preserved.', '',
            '| City | Conservative minus | Metric | Difference [95% interval] |',
            '| --- | --- | --- | --- |']
    for r in paired:
        if r['bootstrap_block_days']==7:
            lines.append(f"| {r['city']} | {r['second']} | {r['metric']} | {r['mean_difference']:.2f} "
                         f"[{r['ci_low']:.2f}, {r['ci_high']:.2f}] |")
    lines+=['', '## Interpretation by city', '']
    for code in ('NYC','CHI'):
        table={r['variant']:r for r in summaries if r['city']==code}
        a,b=table['conservative_truncation'],table['current_truncation']
        effect=a['worst_5pct_mean_pnl']-b['worst_5pct_mean_pnl']
        mean=a['mean_pnl']-b['mean_pnl']
        lines.append(f"- {code}: observed tail-mean change {effect:+.2f}/day; average net-PnL change {mean:+.2f}/day "
                     f"versus current truncation. Heavy winner discounts: {b['incorrectly_heavily_discounted_winner_days']} "
                     f"to {a['incorrectly_heavily_discounted_winner_days']}.")
    lines+=['', 'Chicago has 69 common test dates and four dates in each worst-5% tail. Its margin is 1F '
            'from 42 training dates through May 30, with common b=100 and hypothetical fee=$0.05/share. '
            'Five of the 74 test dates fail the fixed minute-coverage rule. Unlike NYC, Chicago has no '
            'heavily discounted winning bins under either truncation rule, and its worst day is unchanged. '
            'It therefore does not independently reproduce the rare sensor-error failure or demonstrate '
            'that the safeguard prevents it. The observed tail improvement is small and its interval includes zero.', '',
            'No city was dropped or replaced based on results. No pooled two-city significance claim is made. '
            'Report gains, losses and intervals together. Different city-specific b and fees mean dollar levels '
            'should be compared within cities, not interpreted as a common-capital cross-city ranking.', '',
            'Net PnL includes hypothetical maker fee income. Counterfactual replay assumes observed traders '
            'would offer the same trades under different quotes. Live publication delay is a scenario proxy. '
            'The empirical training quantile is not a guaranteed bound on future sensor errors.', '']
    (OUT/'replication_report.md').write_text('\n'.join(lines),encoding='utf-8')


def main():
    protocol=verify_protocol()
    conservative.run_city(protocol['replication_city'],1000)
    manifest_path=OUT/protocol['replication_city']/'manifest.json'
    manifest=json.loads(manifest_path.read_text())
    manifest['confirmation_status']='frozen-rule out-of-city replication; earlier unguarded CHI aggregates visible; not fully blind'
    manifest['replication_protocol_sha256']=hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
    manifest['replication_completed_at_utc']=dt.datetime.now(dt.timezone.utc).isoformat()
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    summaries,paired=[],[]
    for code in ('NYC','CHI'):
        a,b=compare_city(code)
        summaries.extend(a);paired.extend(b)
    write_csv('results/conservative_truncation/replication_summary.csv',summaries)
    write_csv('results/conservative_truncation/replication_paired_intervals.csv',paired)
    write_report(protocol,summaries,paired)
    print((OUT/'replication_report.md').read_text(),flush=True)


if __name__=='__main__':
    main()
