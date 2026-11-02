"""Multi-source ML forecast of the Central Park high, turned into probabilities for the six bins.

Run after collect_features.py. Every forecast is out of sample: the model is refit weekly by
walk-forward, and a forecast for day D is fit only on days up to D-2 (the D-1 high is not final at
the noon cutoff). Training history starts in March 2024, before the Kalshi window, because the
target is the station high and needs no market.

Model: a random forest predicts the residual of the high from the mean of the four point forecasts
(ECMWF, NBM, MAV, MET), so it learns corrections instead of extrapolating raw temperatures. The
forest was chosen after comparing ridge regression, gradient boosting and an average of the three
(see DATA_NOTES.md): ridge scored best on train days but overfit the summer test period. The spread
comes from a second fit: log sd is linear in source disagreement, the NBM spread and season, fit by
maximum likelihood on out-of-fold residuals. The plain forecast average ("consensus") is kept as a
reference with its own fitted spread.

Writes data/ml_forecasts.csv (per contract), data/ml_daily.csv (per day) and, with --ablation,
data/results/ml/ablation.csv (walk-forward refits without each feature group).
"""
import argparse
import json
import math

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline

import config
import forecasts
from collect_nyc import DATA, write_csv

SOURCES = ['ecmwf', 'nbm', 'mav', 'met']
REFIT_DAYS = 7
GAP_DAYS = 2  # Verified highs are known through D-2 at the cutoff.
BIAS_WINDOW = 14
MIN_SD = 1.0
OUT = DATA / 'results' / 'ml'

# Feature groups, used both to build the matrix and for ablation.
GROUPS = {
    'ecmwf': ['dev_ecmwf', 'ecmwf_t850_anom', 'ecmwf_dewpoint_18z_f', 'ecmwf_mslp_18z_hpa', 'ecmwf_night_max'],
    'nbm': ['dev_nbm', 'nbm_max_sd_f', 'nbm_hourly_gap', 'nbm_sky_pct', 'nbm_precip_prob_12h', 'nbm_solar_18z',
            'nbm_dewpoint_18z_f', 'nbm_wind_u_18z', 'nbm_wind_v_18z'],
    'gfs_mos': ['dev_mav', 'mav_cloud_18z', 'mav_precip_prob_12h', 'mav_wind_u_18z', 'mav_wind_v_18z'],
    'nam_mos': ['dev_met', 'met_cloud_18z', 'met_precip_prob_12h'],
    'observations': ['obs_vs_baseline', 'obs_max_so_far_anom', 'obs_temp_change_24h', 'obs_dewpoint_f', 'obs_wind_u_kt',
                     'obs_wind_v_kt', 'obs_mslp_hpa', 'obs_mslp_change_12h', 'obs_sky_cover',
                     'coast_gradient', 'lga_minus_nyc'],
    'recent_errors': [f'bias_{s}' for s in SOURCES] + ['bias_baseline', 'mae_baseline'],
    'season_persistence': ['doy_sin', 'doy_cos', 'baseline_anom', 'persistence_vs_baseline'],
    'disagreement': ['source_sd', 'source_range'],
}
MODELS = ('ml', 'consensus')
LABEL = {'ml': 'ML random forest', 'consensus': 'Forecast consensus'}


# ---------------------------------------------------------------- features

def normals():
    """Day-of-year 1991-2020 mean high, smoothed over +/-15 days."""
    h = pd.read_csv(DATA / 'historical_tmax.csv', parse_dates=['date'])
    h = h[(h['date'].dt.year >= 1991) & (h['date'].dt.year <= 2020) & h['quality_flag'].isna()]
    by_doy = h.groupby(h['date'].dt.dayofyear.clip(upper=365))['tmax_f'].mean().reindex(range(1, 366)).interpolate()
    wrapped = pd.concat([by_doy.iloc[-15:], by_doy, by_doy.iloc[:15]])
    return wrapped.rolling(31, center=True).mean().iloc[15:-15]


def wind_components(direction, speed):
    rad = np.radians(direction)
    return -speed * np.sin(rad), -speed * np.cos(rad)


def build_frame():
    f = pd.read_csv(DATA / 'features' / 'daily_features.csv', parse_dates=['target_date']).set_index('target_date')
    f = f.asfreq('D')  # Missing days become blank rows so rolling windows are calendar-based.
    for s in SOURCES:
        f[f'{s}_max_f'] = f[f'{s}_max_f'].astype(float)
    point = f[[f'{s}_max_f' for s in SOURCES]]
    f['baseline'] = point.mean(axis=1)
    f['source_sd'] = point.std(axis=1)
    f['source_range'] = point.max(axis=1) - point.min(axis=1)
    for s in SOURCES:
        f[f'dev_{s}'] = f[f'{s}_max_f'] - f['baseline']
    normal = normals()
    doy = np.minimum(f.index.dayofyear.to_numpy(), 365)
    f['normal_f'] = normal.reindex(doy).to_numpy()
    f['doy_sin'], f['doy_cos'] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    f['baseline_anom'] = f['baseline'] - f['normal_f']
    f['persistence_vs_baseline'] = f['ghcn_high_d2_f'] - f['baseline']
    f['ecmwf_t850_anom'] = f['ecmwf_t850_18z_f'] - f['baseline']
    f['ecmwf_night_max'] = (f['ecmwf_max_step_hour'] >= 51).astype(float).where(f['ecmwf_max_step_hour'].notna())
    f['nbm_hourly_gap'] = f['nbm_hourly_max_f'] - f['nbm_max_f']
    for m in ('nbm', 'mav'):
        f[f'{m}_wind_u_18z'], f[f'{m}_wind_v_18z'] = wind_components(f[f'{m}_wind_dir_18z'], f[f'{m}_wind_kt_18z'])
    f['obs_vs_baseline'] = f['nyc_temp_f'] - f['baseline']
    f['obs_max_so_far_anom'] = f['obs_max_so_far_f'] - f['normal_f']
    f['coast_gradient'] = f['ewr_temp_f'] - f['jfk_temp_f']  # Inland minus ocean-side: sea-breeze strength.
    f['lga_minus_nyc'] = f['lga_temp_f'] - f['nyc_temp_f']
    # Recent source errors through D-2 only.
    for s in SOURCES + ['baseline']:
        column = 'baseline' if s == 'baseline' else f'{s}_max_f'
        error = (f['high_f'] - f[column]).shift(GAP_DAYS)
        f[f'bias_{s}'] = error.rolling(BIAS_WINDOW, min_periods=5).mean()
        if s == 'baseline':
            f['mae_baseline'] = error.abs().rolling(BIAS_WINDOW, min_periods=5).mean()
    gefs = forecasts.ensemble_moments()  # Window days only; a point-forecast reference in ml_compare.py, not a feature.
    f['gefs_mean_f'] = pd.Series({pd.Timestamp(d): m for d, (m, _) in gefs.items()})
    f['residual'] = f['high_f'] - f['baseline']
    return f[f['baseline'].notna() & f['high_f'].notna()]


def feature_list(drop=()):
    return [c for g, cols in GROUPS.items() if g not in drop for c in cols]


# ---------------------------------------------------------------- model

def make_model():
    return make_pipeline(SimpleImputer(strategy='median'),
                         RandomForestRegressor(n_estimators=400, min_samples_leaf=8, max_features=0.33,
                                               random_state=config.BOOTSTRAP_SEED, n_jobs=-1))


def sigma_design(frame):
    return np.column_stack([np.ones(len(frame)), np.log1p(frame['source_sd'].fillna(frame['source_sd'].median())),
                            np.log1p(frame['nbm_max_sd_f'].fillna(2.0)), frame['doy_sin'], frame['doy_cos']])


def fit_sigma(frame, residuals):
    """Gaussian maximum likelihood for log sd = design @ beta."""
    x = sigma_design(frame)
    r2 = residuals ** 2

    def nll(beta):
        log_sd = x @ beta
        return float(np.sum(log_sd + r2 / (2 * np.exp(2 * log_sd))))

    start = np.zeros(x.shape[1])
    start[0] = math.log(max(np.std(residuals), MIN_SD))
    return minimize(nll, start, method='BFGS').x


def predict_sigma(beta, frame):
    return np.maximum(MIN_SD, np.exp(sigma_design(frame) @ beta))


def fit_predict(train, test, features):
    """Mean and sd for test rows from the forest and the consensus; spreads use 5-fold OOF residuals."""
    x, y = train[features], train['residual'].to_numpy()
    oof = np.zeros(len(train))
    for fit_idx, held_idx in KFold(5).split(x):
        oof[held_idx] = make_model().fit(x.iloc[fit_idx], y[fit_idx]).predict(x.iloc[held_idx])
    corrections = {'ml': (make_model().fit(x, y).predict(test[features]), oof),
                   'consensus': (np.zeros(len(test)), np.zeros(len(train)))}
    return {name: (test['baseline'].to_numpy() + mean, predict_sigma(fit_sigma(train, y - fitted), test))
            for name, (mean, fitted) in corrections.items()}


def walk_forward(frame, features, start, end):
    """Weekly refits; each block of target days is fit on days ending GAP_DAYS before the block."""
    results = []
    block = pd.Timestamp(start)
    while block <= pd.Timestamp(end):
        stop = block + pd.Timedelta(days=REFIT_DAYS)
        test = frame[(frame.index >= block) & (frame.index < stop) & (frame.index <= pd.Timestamp(end))]
        train = frame[frame.index <= block - pd.Timedelta(days=GAP_DAYS)]
        if len(test):
            for name, (mu, sd) in fit_predict(train, test, features).items():
                results.append(pd.DataFrame({'target_date': test.index, 'model': name, 'mean_f': mu, 'sd_f': sd,
                                             'high_f': test['high_f'].to_numpy(), 'train_days': len(train)}))
        block = stop
    out = pd.concat(results, ignore_index=True)
    out['split'] = out['target_date'].map(lambda t: config.split_of(t.date()))
    return out


# ---------------------------------------------------------------- probabilities and scores

def contract_probabilities(daily):
    groups = forecasts.day_groups()
    rows = []
    for r in daily.itertuples():
        day = r.target_date.date().isoformat()
        if day not in groups:
            continue
        group = groups[day]
        p = forecasts.floor_probabilities(forecasts.gaussian_bins(group, r.mean_f, r.sd_f))
        for row, prob in zip(group, p):
            rows.append({'target_date': day, 'market_ticker': row['market_ticker'], 'model': r.model,
                         'probability': prob, 'outcome': int(row['outcome'])})
    return pd.DataFrame(rows)


def mean_scores(probs):
    """Mean Brier and log loss per contract, by split and model."""
    p = probs['probability'].clip(1e-6, 1 - 1e-6)
    probs = probs.assign(brier=(probs['probability'] - probs['outcome']) ** 2,
                         log_loss=-(probs['outcome'] * np.log(p) + (1 - probs['outcome']) * np.log(1 - p)),
                         split=probs['target_date'].map(config.split_of))
    return probs.groupby(['split', 'model'])[['brier', 'log_loss']].mean().reset_index()


def ablation(frame, features):
    """Walk-forward refits without each feature group, and with single sources only."""
    runs = {'all features': features, **{f'without {g}': feature_list(drop=(g,)) for g in GROUPS},
            **{f'only {g}': GROUPS[g] + ['source_sd'] for g in ('nbm', 'ecmwf', 'observations')}}
    rows = []
    for label, cols in runs.items():
        run = walk_forward(frame, cols, config.WINDOW_START, config.WINDOW_END)
        run = run[run['model'] == 'ml']
        scores = mean_scores(contract_probabilities(run)).set_index('split')
        for split in ('train', 'test'):
            err = run[run['split'] == split]
            rows.append({'run': label, 'split': split, 'features': len(cols), 'brier': scores.at[split, 'brier'],
                         'log_loss': scores.at[split, 'log_loss'], 'mae_f': float((err['high_f'] - err['mean_f']).abs().mean())})
        print(f'  {label}: test log loss {rows[-1]["log_loss"]:.4f}', flush=True)
    write_csv('results/ml/ablation.csv', rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ablation', action='store_true', help='also refit without each feature group (slow)')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    frame = build_frame()
    features = feature_list()
    print(f'{len(frame)} days with a forecast consensus and a verified high; {len(features)} features')

    daily = walk_forward(frame, features, config.WINDOW_START, config.WINDOW_END)
    daily['error_f'] = daily['high_f'] - daily['mean_f']
    write_csv('ml_daily.csv', daily.assign(target_date=daily['target_date'].dt.date.astype(str)).to_dict('records'))
    probs = contract_probabilities(daily)
    wide = probs.pivot_table(index=['target_date', 'market_ticker'], columns='model', values='probability').reset_index()
    wide = wide.rename(columns={m: f'{m}_probability' for m in MODELS})
    write_csv('ml_forecasts.csv', wide[['target_date', 'market_ticker', 'ml_probability', 'consensus_probability']].to_dict('records'))
    scores = mean_scores(probs)
    print(scores.to_string(index=False))
    (OUT / 'model.json').write_text(json.dumps({
        'model': LABEL['ml'], 'features': features, 'refit_days': REFIT_DAYS, 'gap_days': GAP_DAYS,
        'history_start': str(frame.index.min().date()), 'scores': scores.to_dict('records')}, indent=2))
    if args.ablation:
        ablation(frame, features)


if __name__ == '__main__':
    main()
