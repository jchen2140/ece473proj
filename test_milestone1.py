import datetime as dt
import math
import unittest

from milestone1 import STATION, parse_tmax, score, seasonal_distance, training_sample
from collect_nyc import hits


class MilestoneChecks(unittest.TestCase):
    def test_quality_flags_missing_values_and_leap_day(self):
        fields = ['-9999   '] * 31
        fields[0] = '  200  A'  # 20 C, accepted source A
        fields[1] = '  201 QA'  # quality flag Q must be rejected
        fields[28] = '  210  A'  # February 29 exists in 2020
        rows, rejected = parse_tmax(f'{STATION}202002TMAX' + ''.join(fields))
        self.assertEqual([r['date'] for r in rows], ['2020-02-01', '2020-02-29'])
        self.assertEqual(rows[0]['tmax_f'], 68)
        self.assertEqual(rejected, 27)

    def test_seasonal_window_wraps_year_and_excludes_future(self):
        self.assertEqual(seasonal_distance(dt.date(1991,12,31),dt.date(2026,1,1)),1)
        history = [{'date': d, 'ghcn_station_id': STATION, 'quality_flag': '',
                    'tmax_f': 80, 'rounded_tmax_f': 80}
                   for d in ['1990-07-20','1991-07-05','2020-08-05','2020-07-20','2026-07-20']]
        sample = training_sample(history, dt.date(2026,7,20))
        self.assertEqual([r['date'] for r in sample], ['1991-07-05','2020-07-20'])

    def test_bin_boundaries_partition_integer_temperatures(self):
        bins = [{'strike_type':'less','cap_strike':77},
                {'strike_type':'between','floor_strike':77,'cap_strike':78},
                {'strike_type':'greater','floor_strike':78}]
        for temperature in [50,76,77,78,79,100]:
            self.assertEqual(sum(hits(b,temperature) for b in bins),1)

    def test_scoring_zero_probabilities_and_perfect_forecast(self):
        perfect=score([0,1],[0,1])
        self.assertEqual(perfect['brier_mean_per_binary_contract'],0)
        wrong=score([0,1],[1,0])
        self.assertEqual(wrong['brier_mean_per_binary_contract'],1)
        self.assertTrue(math.isfinite(wrong['binary_log_loss_mean']))
        with self.assertRaises(ValueError):
            score([1.1],[0])


if __name__ == '__main__':
    unittest.main()
