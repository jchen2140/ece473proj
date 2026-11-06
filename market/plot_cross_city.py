"""Plot paired conservative-minus-current effects from the frozen-rule replication."""
import csv

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from market.cross_city import OUT


def main():
    with (OUT/'replication_paired_intervals.csv').open(newline='') as f:
        rows=[r for r in csv.DictReader(f) if r['second']=='current_truncation'
              and int(r['bootstrap_block_days'])==7]
    fig,axes=plt.subplots(1,2,figsize=(10,4.3),sharey=True)
    for ax,metric,title in zip(axes,('mean_net_pnl','worst_5pct_mean_pnl'),
                              ('Average net PnL','Mean PnL on worst 5% of days')):
        for y,city in enumerate(('NYC','CHI')):
            r=next(r for r in rows if r['city']==city and r['metric']==metric)
            value,low,high=[float(r[k]) for k in ('mean_difference','ci_low','ci_high')]
            ax.hlines(y,low,high,color='#276678',linewidth=2.5)
            ax.plot(value,y,'o',color='#153d4a',markersize=7)
            ax.annotate(f'{value:+.1f} [{low:+.1f}, {high:+.1f}]',(value,y),
                        xytext=(0,12),textcoords='offset points',ha='center',fontsize=9)
        ax.axvline(0,color='#888888',linestyle='--',linewidth=1)
        ax.set_title(title,fontsize=12)
        ax.set_xlabel('Conservative minus current truncation ($/day)')
        ax.set_yticks([0,1],['NYC','Chicago'])
        ax.set_ylim(-.5,1.6)
        ax.grid(axis='x',alpha=.15)
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Does a sensor-error safety margin transfer to another city?',fontsize=14)
    fig.text(.5,.015,'Paired 95% weekly-block intervals; positive favors conservative truncation.\n'
             'NYC: exploratory design sample. Chicago: frozen-rule replication. Hypothetical maker fees.',
             ha='center',fontsize=9,color='#444444')
    fig.tight_layout(rect=(0,.12,1,.94))
    path=OUT/'replication_comparison.png'
    fig.savefig(path,dpi=180,bbox_inches='tight')
    print(path)


if __name__=='__main__':
    main()
