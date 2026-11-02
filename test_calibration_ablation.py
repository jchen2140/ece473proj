import datetime as dt
import math
import unittest

import numpy as np

import ablation_replay
import calibration_ablation as experiment
from lmsr import LMSR


def group():
    bounds = [('',77,'less'),(77,78,'between'),(79,80,'between'),
              (81,82,'between'),(83,84,'between'),(84,'','greater')]
    return [{'market_ticker':f'BIN{i}','lower_bound':str(lo),'upper_bound':str(hi),
             'strike_type':kind,'outcome':int(i==2),'ensemble_probability':(1 if i==2 else 0),
             'prediction_time_utc':'2026-06-29T16:00:00+00:00'}
            for i,(lo,hi,kind) in enumerate(bounds)]


def series(first=dt.date(2026,5,1),days=70):
    moments,highs = {},{}
    for i in range(days):
        date = (first+dt.timedelta(days=i)).isoformat()
        moments[date] = (80+i%4,4+i%3)
        highs[date] = moments[date][0]-3+(i%5-2)*0.2
    return moments,highs


class ControlledCalibration(unittest.TestCase):
    def setUp(self):
        self.moments,self.highs = series()
        self.day = '2026-06-30'
        self.settings = {'window':30,'spread_method':'scaled'}

    def test_all_variants_partition_bins_and_share_probability_floor(self):
        fitted,ps = experiment.predictions(self.day,group(),self.moments,self.highs,self.settings)
        self.assertEqual(set(ps),set(experiment.VARIANTS))
        for values in ps.values():
            self.assertAlmostEqual(sum(values),1)
            self.assertGreaterEqual(min(values),0.02/6-1e-12)
        self.assertAlmostEqual(ps['empirical_smoothed'][2],0.98+0.02/6)

    def test_mean_and_spread_switches_are_independent(self):
        f = experiment.fits(self.day,self.moments,self.highs,30,'scaled')
        self.assertEqual(f['bias_only'][1],f['gaussian_uncorrected'][1])
        self.assertEqual(f['spread_only'][0],f['gaussian_uncorrected'][0])
        self.assertEqual(f['bias_and_spread'][0],f['bias_only'][0])
        self.assertEqual(f['bias_and_spread'][1],f['spread_only'][1])
        self.assertNotEqual(f['bias_only'][0],f['gaussian_uncorrected'][0])

    def test_previous_day_target_and_future_outcomes_do_not_change_prediction(self):
        before = experiment.predictions(self.day,group(),self.moments,self.highs,self.settings)
        changed = {**self.highs,'2026-06-29':500,'2026-06-30':500,'2026-07-01':500}
        after = experiment.predictions(self.day,group(),self.moments,changed,self.settings)
        self.assertEqual(before,after)

    def test_warmup_returns_no_prediction_for_all_variants_together(self):
        self.assertIsNone(experiment.predictions('2026-05-10',group(),self.moments,self.highs,self.settings))

    def test_hyperparameter_selection_ignores_test_outcomes(self):
        moments,highs = series(dt.date(2025,8,14),80)
        moments['2026-06-30'] = (90,4)
        highs['2026-06-30'] = 90
        groups = {'2025-09-20':group(),'2026-06-30':group()}
        before,_ = experiment.select_settings(groups,moments,highs)
        highs['2026-06-30'] = -100
        after,_ = experiment.select_settings(groups,moments,highs)
        self.assertEqual(before,after)
        self.assertEqual(before['training_days'],1)
        self.assertEqual(before['training_end'],'2025-09-20')

    def test_export_preserves_cutoff_and_calibration_provenance(self):
        rows,daily,excluded = experiment.export_predictions({self.day:group()},self.moments,self.highs,self.settings)
        self.assertEqual((len(rows),len(daily),len(excluded)),(30,5,0))
        for row in rows:
            self.assertEqual(row['prediction_time_utc'],'2026-06-29T16:00:00+00:00')
            if row['calibration_observation_count']:
                self.assertEqual(row['latest_calibration_observation'],'2026-06-28')
                self.assertEqual(row['calibration_observation_count'],30)
            else:
                self.assertEqual(row['latest_calibration_observation'],'')


class UncertaintyAndRisk(unittest.TestCase):
    def test_block_resampling_is_seeded_bounded_and_contiguous(self):
        draws = experiment.bootstrap_indices(21,20,7,42)
        self.assertEqual(draws.shape,(20,21))
        self.assertTrue(np.array_equal(draws,experiment.bootstrap_indices(21,20,7,42)))
        self.assertTrue(np.all((draws>=0)&(draws<21)))
        for start in (0,7,14):
            self.assertTrue(np.all(np.diff(draws[:,start:start+7],axis=1)==1))
        zero = experiment.interval(np.zeros(21),draws)
        self.assertEqual(zero,{'mean':0.0,'ci_low':0.0,'ci_high':0.0})

    def test_equal_budget_maps_different_priors_to_same_loss_bound(self):
        for ps in ([1/6]*6,[0.01,0.09,0.1,0.2,0.3,0.3]):
            b = ablation_replay.liquidity_for_budget(ps,300)
            self.assertAlmostEqual(LMSR(ps,b).loss_bound(),300)
        with self.assertRaises(ValueError):
            ablation_replay.liquidity_for_budget([0,1],300)
        with self.assertRaises(ValueError):
            ablation_replay.liquidity_for_budget([0.5,0.5],math.nan)

    def test_replay_retains_same_budget_and_accounts_for_fees(self):
        ps = [1/6]*6
        records = [{'day':'2026-06-01','winner':0,'seeds':{m:ps for m in ablation_replay.MODELS}}]
        flow = {'2026-06-01':[(0.0,0,True,0.4,100.0),(1.0,1,True,0.4,100.0)]}
        rows = ablation_replay.run(records,flow,300,300,[0.0,0.05])
        for row in rows:
            self.assertAlmostEqual(row['pnl'],row['trading_pnl']+row['fee_income'])
            if row['risk_mode']=='equal_loss_budget':
                self.assertAlmostEqual(row['opening_loss_bound'],300)
            self.assertGreaterEqual(row['trading_pnl'],-row['opening_loss_bound']-1e-8)
        # Identical seeds receive identical results: comparison does not depend on variant label.
        for mode in ('equal_loss_budget','fixed_b'):
            for fee in (0.0,0.05):
                selected = [r for r in rows if r['risk_mode']==mode and r['fee']==fee]
                self.assertEqual(len({round(r['pnl'],10) for r in selected}),1)


if __name__=='__main__':
    unittest.main()
