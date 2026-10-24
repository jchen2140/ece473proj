import datetime as dt
import unittest

from climatology import SeasonalClimatology, quantile, seasonal_distance


def record(date, high=80, quality=''):
    return {'date': date, 'ghcn_station_id': 'USW00094728', 'tmax_f': high,
            'rounded_tmax_f': int(high+0.5), 'quality_flag': quality}


class ClimatologyChecks(unittest.TestCase):
    def test_future_records_do_not_change_prediction(self):
        training = [record('2019-07-20', 80), record('2020-07-20', 90)]
        a = SeasonalClimatology(training, start_year=2019)
        b = SeasonalClimatology(training+[record('2026-07-20', 120)], start_year=2019)
        target = dt.date(2026,7,20)
        self.assertEqual(a.distribution(target), b.distribution(target))
        self.assertEqual(a.probability(target, {'strike_type': 'between','floor_strike':80,'cap_strike':89}),0.5)

    def test_predictions_inside_training_period_are_rejected(self):
        model = SeasonalClimatology([record('2020-07-20')])
        with self.assertRaises(ValueError):
            model.sample(dt.date(2020,7,20))

    def test_leap_day_window_and_year_wrap(self):
        self.assertEqual(seasonal_distance(dt.date(2000,2,28),dt.date(2024,3,1)),2)
        model = SeasonalClimatology([record('2019-12-31'),record('2020-01-01',82)], window_days=1)
        self.assertEqual(len(model.sample(dt.date(2026,1,1))),2)

    def test_cdf_is_monotone_and_distribution_sums_to_one(self):
        model = SeasonalClimatology([record('2020-07-19',80),record('2020-07-20',80),record('2020-07-21',90)])
        target = dt.date(2026,7,20)
        self.assertAlmostEqual(sum(r['probability'] for r in model.distribution(target)),1)
        cdf = [model.cdf(target,x) for x in [79,80,89,90,100]]
        self.assertEqual(cdf,sorted(cdf))
        self.assertEqual((cdf[0],cdf[-1]),(0,1))
        self.assertAlmostEqual(cdf[1],2/3)

    def test_quality_duplicates_and_empty_windows_are_rejected(self):
        for records in [[record('2020-07-20',quality='X')], [record('2020-07-20')]*2]:
            with self.assertRaises(ValueError):
                SeasonalClimatology(records)
        with self.assertRaises(ValueError):
            SeasonalClimatology([record('2020-01-01')]).sample(dt.date(2026,7,20))

    def test_quantiles_include_endpoints_and_interpolate(self):
        self.assertEqual(quantile([70,90],0),70)
        self.assertEqual(quantile([70,90],0.5),80)
        self.assertEqual(quantile([70,90],1),90)


if __name__ == '__main__':
    unittest.main()
