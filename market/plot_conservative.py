"""Plot saved conservative-truncation results without replaying trades."""
import argparse
import csv
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from market.conservative import OUT,VARIANTS


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--city',default='NYC')
    args=parser.parse_args()
    folder=OUT/args.city
    manifest=json.loads((folder/'manifest.json').read_text())
    with (folder/'summary.csv').open() as f:
        rows={r['variant']:r for r in csv.DictReader(f) if r['bootstrap_block_days']=='7'}
    fig,axes=plt.subplots(1,3,figsize=(12,4.1))
    labels=['Forecast only','Current truncation','Conservative truncation']
    colors=['#778899','#ae7657','#2b7166']
    y=np.arange(3)
    for ax in axes:
        ax.set_yticks(y,labels)
        ax.set_ylim(2.7,-.7)
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='x',alpha=.15)
        ax.set_axisbelow(True)
    for i,v in enumerate(VARIANTS):
        r=rows[v]
        axes[0].plot([float(r['ci_low']),float(r['ci_high'])],[i,i],color=colors[i],linewidth=2)
        axes[0].scatter(float(r['mean_pnl']),i,color=colors[i],s=40,zorder=3)
    axes[0].axvline(0,color='#888888',linestyle='--',linewidth=1)
    axes[0].set(title='Average net PnL',xlabel='Dollars/day · 95% weekly-block interval')
    axes[1].barh(y,[float(rows[v]['worst_5pct_mean_pnl']) for v in VARIANTS],color=colors)
    axes[1].set(title='Downside performance',xlabel='Mean PnL on worst 5% of days ($)')
    values=[int(rows[v]['incorrectly_heavily_discounted_winner_days']) for v in VARIANTS]
    axes[2].barh(y,values,color=colors)
    for i,v in enumerate(values):axes[2].text(v+.02,i,str(v),va='center')
    axes[2].set(title='Incorrectly discounted winners',xlabel='Days with winner weight reduced to 0.001',xlim=(0,max(values)+.4))
    for ax in axes[1:]:ax.set_yticklabels([])
    fig.suptitle('Conservative truncation: the risk–return tradeoff',fontsize=16,x=.035,ha='left',y=.98)
    fig.text(.035,.86,f"{args.city} · {manifest['test_days']} common test days · {manifest['margin']['margin_f']:g}°F training-only margin · shared b={manifest['common_settings']['b0']} · fee={manifest['common_settings']['fee']:.2f}/share",fontsize=10)
    fig.text(.035,.035,'Exploratory counterfactual replay; hypothetical maker fee. The known NYC July 31 loss remains included.\n'
             'Missing observation dates are audited. No independent confirmation or unconditional sensor-error guarantee.',fontsize=9,color='#555555')
    fig.tight_layout(rect=(0,.14,1,.86),w_pad=2)
    fig.savefig(folder/'comparison.png',dpi=180,bbox_inches='tight')
    plt.close(fig)


if __name__=='__main__':
    main()
