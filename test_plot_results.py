import unittest

import plot_results


class ReliabilityAndDaily(unittest.TestCase):
    def rows(self):
        return [
            {'target_date': '2026-04-30', 'split': 'train', 'outcome': '1',
             'ensemble_probability': '0.05', 'climatology_probability': '0.2',
             'persistence_probability': '0.1', 'market_midpoint': '0.4'},
            {'target_date': '2026-04-30', 'split': 'train', 'outcome': '0',
             'ensemble_probability': '0.95', 'climatology_probability': '0.8',
             'persistence_probability': '0.9', 'market_midpoint': '0.6'},
            {'target_date': '2026-05-01', 'split': 'test', 'outcome': '1',
             'ensemble_probability': '0.25', 'climatology_probability': '0.25',
             'persistence_probability': '0.25', 'market_midpoint': '0.25'},
            {'target_date': '2026-05-01', 'split': 'test', 'outcome': '0',
             'ensemble_probability': '0.75', 'climatology_probability': '0.75',
             'persistence_probability': '0.75', 'market_midpoint': '0.75'},
        ]

    def test_complete_rows_drop_blank_models(self):
        rows = self.rows()
        rows[0]['market_midpoint'] = ''
        kept = plot_results.complete_rows(rows, ['raw_ensemble', 'market'])
        self.assertEqual(len(kept), 3)

    def test_reliability_bins_match_milestone1_edges(self):
        rows = [{'ensemble_probability': str(p), 'outcome': '1' if p >= 0.5 else '0'}
                for p in (0.0, 0.09, 0.1, 0.99, 1.0)]
        bins = plot_results.reliability_bins(rows, 'raw_ensemble')
        counts = {(round(b['bin_lower'], 1), round(b['bin_upper'], 1)): b['count'] for b in bins}
        self.assertEqual(counts[(0.0, 0.1)], 2)
        self.assertEqual(counts[(0.1, 0.2)], 1)
        self.assertEqual(counts[(0.9, 1.0)], 2)

    def test_daily_scores_average_the_six_contract_brier(self):
        daily = plot_results.daily_scores(self.rows(), ['raw_ensemble'])
        self.assertEqual([d['target_date'] for d in daily], ['2026-04-30', '2026-05-01'])
        self.assertAlmostEqual(daily[0]['raw_ensemble_brier'], ((0.05 - 1) ** 2 + (0.95 - 0) ** 2) / 2)
        self.assertEqual(daily[1]['split'], 'test')

    def test_monthly_scores_pool_by_year_month(self):
        monthly = plot_results.monthly_scores(plot_results.daily_scores(self.rows(), ['raw_ensemble']))
        self.assertEqual([m['month'] for m in monthly], ['2026-04', '2026-05'])
        self.assertEqual(monthly[0]['days'], 1)


if __name__ == '__main__':
    unittest.main()
