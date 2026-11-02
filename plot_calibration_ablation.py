"""Plot the saved controlled forecast/replay experiment; never recollects data."""
import csv
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from calibration_ablation import OUT


def read(name):
    with (OUT/name).open(newline='',encoding='utf-8') as f:
        return list(csv.DictReader(f))


def main():
    manifest = json.loads((OUT/'replay_manifest.json').read_text())
    fee = max(manifest['fees'])
    models = ['empirical_smoothed','gaussian_uncorrected','bias_only','spread_only','bias_and_spread','kalshi']
    labels = ['Empirical GEFS','Gaussian GEFS','Bias only','Spread only','Bias + spread','Kalshi benchmark']
    colors = ['#a4adb8','#778b9e','#478f9d','#ba9645','#27634e','#353c46']
    summary = {r['forecast_variant']:r for r in read('replay_summary.csv')
               if r['risk_mode']=='equal_loss_budget' and float(r['fee'])==fee and r['bootstrap_block_days']=='7'}
    daily = [r for r in read('replay_daily.csv') if r['risk_mode']=='equal_loss_budget' and float(r['fee'])==fee]
    y = np.arange(len(models))
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes = plt.subplots(2,2,figsize=(12.5,9.2))
    for ax in axes.flat:
        ax.set_yticks(y,labels)
        ax.invert_yaxis()
        ax.grid(axis='x',alpha=0.15)
        ax.set_axisbelow(True)
    ax=axes[0,0]
    ax.barh(y,[float(summary[m]['opening_brier']) for m in models],color=colors,height=.65)
    ax.set(title='A. Forecast accuracy',xlabel='Mean binary Brier score · lower is better',xlim=(0,.165))
    ax=axes[0,1]
    counts=[sum(int(r['very_low_probability_winner']) for r in daily if r['forecast_variant']==m) for m in models]
    ax.barh(y,counts,color=colors,height=.65)
    for i,v in enumerate(counts): ax.text(v+.15,i,str(v),va='center',fontsize=10)
    ax.set(title='B. Low-probability winners',xlabel='Days where the realized winning bin had <1% probability',xlim=(0,max(counts)+2))
    ax=axes[1,0]
    for i,m in enumerate(models):
        s=summary[m]
        ax.plot([float(s['pnl_ci_low']),float(s['pnl_ci_high'])],[i,i],color=colors[i],linewidth=2)
        ax.scatter(float(s['mean_daily_pnl']),i,color=colors[i],s=42,zorder=3)
    ax.axvline(0,color='#999999',linestyle='--',linewidth=1)
    ax.set(title='C. Equal-budget net PnL',xlabel='Dollars/day · 95% intervals from 7-day moving blocks')
    ax=axes[1,1]
    ax.barh(y,[float(summary[m]['worst_5pct_mean_pnl']) for m in models],color=colors,height=.65)
    ax.set(title='D. Downside performance',xlabel='Mean PnL on six worst days ($) · less negative is better')
    fig.suptitle('What does weather calibration improve?',fontsize=19,fontweight='bold',x=.04,ha='left',y=.99)
    fig.text(.04,.945,f"Central Park · {manifest['test_days']} test days · same arrivals · ${manifest['equal_loss_budget']:g} loss budget/event · ${fee:.2f}/share hypothetical maker fee",fontsize=11,parse_math=False)
    fig.text(.04,.022,'Exploratory counterfactual replay. Current test results informed experiment design; independent confirmation is required.\n'
             'Panel B is a retrospective diagnostic, not an available-at-cutoff trading signal. All forecasts share 2% uniform smoothing.',fontsize=9,color='#555555')
    fig.tight_layout(rect=(0,.07,1,.92),h_pad=2.2,w_pad=2.5)
    for extension in ('png','svg'):
        fig.savefig(OUT/f'experiment_overview.{extension}',dpi=180,bbox_inches='tight')
    svg = OUT/'experiment_overview.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8')
    plt.close(fig)
    print('Saved',OUT/'experiment_overview.png')


if __name__=='__main__':
    main()
