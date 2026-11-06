import datetime as dt
import math
import unittest

import numpy as np

from market import conservative, makers
from tests.test_makers import synthetic_day


def audit_rows():
    return [{'day':(dt.date(2026,4,1)+dt.timedelta(days=i)).isoformat(),'split':'train','eligible':1,
             'sensor_minus_official_f':1 if i%3 else .5} for i in range(30)]


class MarginTest(unittest.TestCase):
    def test_margin_uses_only_available_training_days(self):
        rows = audit_rows()
        before = conservative.fit_margin(rows,'2026-04-30')
        after = conservative.fit_margin(rows+[
            {'day':'2026-05-01','split':'train','eligible':1,'sensor_minus_official_f':100},
            {'day':'2026-06-01','split':'test','eligible':1,'sensor_minus_official_f':100}], '2026-04-30')
        self.assertEqual(before,after)
        self.assertEqual(before['margin_f'],1)
        self.assertEqual(before['training_days'],30)

    def test_margin_rounds_up_and_does_not_treat_underread_as_overread(self):
        rows = audit_rows()
        for r in rows:r['sensor_minus_official_f']=-5
        self.assertEqual(conservative.fit_margin(rows,'2026-04-30')['margin_f'],0)
        rows[-1]['sensor_minus_official_f']=1.1
        self.assertEqual(conservative.fit_margin(rows,'2026-04-30')['margin_f'],1.5)

    def test_fit_cutoff_excludes_day_before_first_test(self):
        self.assertEqual(conservative.latest_fit_date([{'day':'2026-06-01','split':'test'}]),'2026-05-30')


class ConservativeTiltTest(unittest.TestCase):
    def test_zero_margin_matches_original_and_positive_margin_preserves_more_weight(self):
        day = synthetic_day(np.random.default_rng(7),5)
        day['running_max']=np.full(len(day['minute_ts']),86.0)
        ts=day['minute_ts'][-1]+makers.FEED_DELAY
        before,_=makers.tilt(day,ts)
        zero,_=conservative.conservative_tilt(day,ts,0)
        guarded,_=conservative.conservative_tilt(day,ts,2)
        np.testing.assert_array_equal(before,zero)
        self.assertTrue(np.all(guarded>=before))

    def test_infinite_upper_tail_cannot_be_knocked_out(self):
        day=synthetic_day(np.random.default_rng(4),5)
        day['running_max']=np.full(len(day['minute_ts']),200.0)
        weights,_=conservative.conservative_tilt(day,day['minute_ts'][-1]+makers.FEED_DELAY,1)
        self.assertEqual(weights[-1],1)

    def test_observation_not_used_before_delay(self):
        day=synthetic_day(np.random.default_rng(4),5)
        weights,reading=conservative.conservative_tilt(day,day['minute_ts'][0]+makers.FEED_DELAY-1,1)
        self.assertIsNone(reading)
        np.testing.assert_array_equal(weights,np.ones(6))

    def test_measurement_error_within_margin_does_not_discount_winning_bin(self):
        day=synthetic_day(np.random.default_rng(4),2)
        official=84.0
        day['running_max']=np.full(len(day['minute_ts']),official+2)
        weights,_=conservative.conservative_tilt(day,day['minute_ts'][-1]+makers.FEED_DELAY,2)
        self.assertEqual(weights[2],1)

    def test_error_beyond_margin_still_has_a_finite_winner_specific_bound(self):
        day=synthetic_day(np.random.default_rng(4),2)
        day.update(day='2026-06-01',split='test')
        day['running_max']=np.full(len(day['minute_ts']),100.0)
        settings={'b0':100,'fee':.05,'sd_ref':2.0}
        r=conservative.run_day(day,'conservative_truncation',settings,1)
        self.assertEqual(r['winner_heavily_discounted'],1)
        expected=100*math.log(1/day['priors']['public'][2])+100*math.log(1000)
        self.assertAlmostEqual(r['winner_specific_bound'],expected)
        self.assertGreaterEqual(r['trading_pnl'],-expected-1e-6)

    def test_callback_default_preserves_original_replay(self):
        day=synthetic_day(np.random.default_rng(4),2)
        a=makers.replay(day,'forecast_live',100,.05,1,2)
        b=makers.replay(day,'forecast_live',100,.05,1,2,tilt_function=makers.tilt)
        self.assertEqual(a,b)


class BootstrapTest(unittest.TestCase):
    def test_weekly_blocks_keep_calendar_gaps_and_all_valid_dates(self):
        days=[f'2026-06-{i:02d}' for i in list(range(1,8))+list(range(10,17))]
        draws=conservative.bootstrap_draws(days,200,7)
        self.assertTrue(np.any(draws==-1))
        self.assertEqual(set(draws[draws>=0]),set(range(len(days))))
        for row in draws:
            for i in (0,7,14):
                selected=[(j,dt.date.fromisoformat(days[k])) for j,k in enumerate(row[i:i+7]) if k>=0]
                self.assertTrue(all((b-a).days==j-i for (i,a),(j,b) in zip(selected,selected[1:])))
        np.testing.assert_allclose(conservative.resampled_means(np.full(len(days),3.0),draws),3)


if __name__=='__main__':
    unittest.main()
