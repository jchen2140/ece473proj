import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from market import cross_city


class TailIntervalsTest(unittest.TestCase):
    def test_tail_uses_ceiling_of_five_percent(self):
        self.assertEqual(cross_city.tail_mean(np.arange(21)), .5)

    def test_identical_paired_strategies_have_zero_interval(self):
        values=np.array([-10., 3., 7.])
        draws=np.array([[0,1,2],[-1,0,2],[2,2,1]])
        result=cross_city.paired_tail_intervals(values,values,draws)
        self.assertEqual(set(result.values()),{0.})

    def test_tail_recomputed_for_each_maker_and_missing_dates_ignored(self):
        # The worst day differs across makers: paired mean differences would be zero.
        a,b=np.array([-100.,20.]),np.array([0.,-80.])
        result=cross_city.paired_tail_intervals(a,b,np.array([[0,1,-1],[0,1,-1]]))
        self.assertEqual(result['mean_difference'],-20)
        self.assertEqual(result['ci_low'],-20)
        self.assertEqual(result['ci_high'],-20)

    def test_invalid_tail_observations_rejected(self):
        for values in ([],[np.nan],[np.inf]):
            with self.assertRaises(ValueError):cross_city.tail_mean(values)

    def test_unpaired_samples_rejected(self):
        with self.assertRaises(ValueError):
            cross_city.paired_tail_intervals([1],[1,2],np.array([[0]]))


class FrozenProtocolTest(unittest.TestCase):
    def test_changed_strategy_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            code=root/'strategy.py'
            code.write_text('original strategy')
            protocol={'code_sha256':{'strategy.py':hashlib.sha256(code.read_bytes()).hexdigest()},
                      'replication_city':'CHI','coverage_fraction':.8}
            path=root/'protocol.json'
            path.write_text(json.dumps(protocol))
            with patch.object(cross_city,'ROOT',root):
                self.assertEqual(cross_city.verify_protocol(path),protocol)
                code.write_text('retuned strategy')
                with self.assertRaisesRegex(ValueError,'Frozen strategy code changed'):
                    cross_city.verify_protocol(path)


if __name__=='__main__':
    unittest.main()
