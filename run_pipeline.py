"""Run the whole study in order, or a slice of it.

    python run_pipeline.py                      # every analysis stage, from data already on disk
    python run_pipeline.py --collect            # also re-download everything first (hours; resumable)
    python run_pipeline.py --only market.makers analysis.latency
    python run_pipeline.py --from market.backtest
    python run_pipeline.py --list

Stages run as `python -m <module>` from the repository root, so each one can also be run by hand.
"""
import argparse
import subprocess
import sys
import time

# (module, extra arguments, what it produces). Order matters: each stage reads what earlier ones wrote.
COLLECT = [
    ('collect.nyc', ['--workers', '32'], 'Kalshi NYC contracts and GEFS ensemble -> data/matched.csv'),
    ('collect.check_nyc', [], 'offline consistency checks -> data/daily_summary.csv'),
    ('collect.trades', ['--workers', '2'], 'every NYC taker trade and hourly candle -> data/flow/'),
    ('collect.features', ['--workers', '8'], 'ECMWF, NWS MOS, ASOS features -> data/features/'),
    ('collect.city_markets', [], '15 cities: quotes, MOS runs, ASOS -> data/cities/'),
    ('collect.realtime', [], '6 cities: 1-minute ASOS and every trade -> data/raw/realtime/'),
    ('collect.city_markets', ['--full-year'], '7 cities, Aug 2025-Aug 2026: quotes, MOS runs, ASOS -> data/cities_full/'),
    ('collect.realtime', ['--full-year'], '7 cities, full year: 1-minute ASOS and every trade -> data/raw/realtime_full/'),
]
ANALYSIS = [
    ('forecast.climatology', [], '1991-2020 seasonal climatology -> data/climatology/'),
    ('forecast.baselines', [], 'climatology probabilities for every contract -> data/matched_with_climatology.csv'),
    ('analysis.evaluate', [], 'GEFS, climatology, persistence, Kalshi scores -> data/results/'),
    ('analysis.plot_results', [], 'train/test and monthly comparisons -> data/results/figures/'),
    ('forecast.noaa_calibration', [], 'rolling GEFS correction -> data/noaa_calibrated.csv'),
    ('forecast.ml_model', [], 'walk-forward random forest -> data/ml_forecasts.csv (add --ablation by hand)'),
    ('market.calibrate_traders', [], 'synthetic trader settings -> data/trader_calibration.json'),
    ('analysis.compare_forecasts', [], 'calibrated NOAA vs Kalshi -> data/results/forecast_comparison/'),
    ('market.backtest', [], 'NYC LMSR replay by opening forecast -> data/results/backtest/'),
    ('market.plot_backtest', [], 'backtest figures'),
    ('analysis.ml_compare', [], 'ML vs Kalshi and NOAA, importance, trading -> data/results/ml/'),
    ('analysis.trader_behavior', [], 'who trades when and who wins -> data/results/traders/'),
    ('analysis.efficiency', [], '15-city information efficiency -> data/results/efficiency/'),
    ('analysis.latency', [], 'minute-level knock-out latency -> data/results/realtime/'),
    ('market.makers', [], 'domain market-maker ladder on 6 cities -> data/results/adaptive_mm/'),
    ('forecast.calibration_ablation', [], 'controlled GEFS bias/spread ablation -> data/results/calibration_ablation/'),
    ('market.ablation_replay', [], 'equal-risk and fixed-b replay of the ablation -> data/results/calibration_ablation/'),
    ('analysis.plot_calibration_ablation', [], 'ablation figure'),
    ('market.conservative', ['--cities', 'NYC'], 'training-only sensor margin, NYC -> data/results/conservative_truncation/'),
    ('market.cross_city', [], 'frozen-rule Chicago replication -> data/results/conservative_truncation/'),
    ('market.plot_cross_city', [], 'replication figure'),
    ('market.full_year', [], '7-city full-year walk-forward ladder -> data/results/full_year/'),
    ('market.risk_frontier', [], 'PnL vs capital as b varies -> data/results/risk_frontier/'),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--collect', action='store_true', help='run the download stages first')
    parser.add_argument('--only', nargs='+', metavar='MODULE', help='run just these stages')
    parser.add_argument('--from', dest='start', metavar='MODULE', help='start at this stage')
    parser.add_argument('--list', action='store_true', help='print the stages and exit')
    args = parser.parse_args()
    stages = (COLLECT if args.collect or args.only else []) + ANALYSIS
    if args.list:
        for module, extra, note in COLLECT + ANALYSIS:
            print(f'{module:28s} {note}')
        return
    if args.only:
        unknown = set(args.only) - {m for m, _, _ in stages}
        if unknown:
            raise SystemExit(f'Unknown stage(s): {", ".join(sorted(unknown))}')
        stages = [s for s in stages if s[0] in args.only]
    if args.start:
        names = [m for m, _, _ in stages]
        if args.start not in names:
            raise SystemExit(f'Unknown stage: {args.start}')
        stages = stages[names.index(args.start):]
    for module, extra, note in stages:
        print(f'\n=== {module}: {note}', flush=True)
        started = time.monotonic()
        result = subprocess.run([sys.executable, '-m', module, *extra])
        if result.returncode:
            raise SystemExit(f'{module} failed with exit code {result.returncode}')
        print(f'=== {module} done in {time.monotonic() - started:.0f}s', flush=True)


if __name__ == '__main__':
    main()
