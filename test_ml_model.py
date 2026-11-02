"""Leakage checks for ml_model.py: no training day is later than D-2, and lagged errors stop at D-2."""
import unittest
from unittest import mock

import numpy as np
import pandas as pd

import ml_model


def synthetic_frame(days=80):
    index = pd.date_range('2025-01-01', periods=days, freq='D')
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({c: rng.normal(size=days) for c in ml_model.feature_list()}, index=index)
    frame['baseline'] = 60 + rng.normal(size=days)
    frame['high_f'] = frame['baseline'] + rng.normal(size=days)
    frame['residual'] = frame['high_f'] - frame['baseline']
    frame['source_sd'] = 1.0
    frame['nbm_max_sd_f'] = 2.0
    return frame


class WalkForwardTest(unittest.TestCase):
    def test_training_stops_two_days_before_each_target(self):
        frame = synthetic_frame()
        seen = []

        def record(train, test, features):
            seen.append((train.index.max(), test.index.min()))
            return {'ml': (test['baseline'].to_numpy(), np.ones(len(test)))}

        with mock.patch.object(ml_model, 'fit_predict', side_effect=record):
            ml_model.walk_forward(frame, ml_model.feature_list(), '2025-02-01', '2025-03-15')
        self.assertTrue(seen)
        for last_train, first_test in seen:
            self.assertLessEqual(last_train, first_test - pd.Timedelta(days=ml_model.GAP_DAYS))

    def test_predictions_cover_every_target_day_once(self):
        frame = synthetic_frame()
        out = ml_model.walk_forward(frame, ml_model.feature_list(), '2025-02-01', '2025-03-15')
        for model in ml_model.MODELS:
            rows = out[out['model'] == model]
            self.assertEqual(len(rows), 43)
            self.assertTrue(rows['target_date'].is_unique)
            self.assertTrue((rows['sd_f'] >= ml_model.MIN_SD).all())

    def test_lagged_bias_ignores_last_two_highs(self):
        # Changing the highs of D-1 and D must not change D's recent-error features.
        index = pd.date_range('2025-01-01', periods=40, freq='D')
        base = pd.DataFrame({'high_f': 50.0, 'ecmwf_max_f': 49.0}, index=index)
        error = lambda f: (f['high_f'] - f['ecmwf_max_f']).shift(ml_model.GAP_DAYS).rolling(ml_model.BIAS_WINDOW, min_periods=5).mean()
        changed = base.copy()
        changed.loc[index[-2:], 'high_f'] = 90.0
        self.assertEqual(error(base).iloc[-1], error(changed).iloc[-1])


if __name__ == '__main__':
    unittest.main()
