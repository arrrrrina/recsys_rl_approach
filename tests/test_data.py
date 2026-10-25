import unittest
import numpy as np
import pandas as pd
from recsys_rl.data import build_examples


class TemporalExamplesTest(unittest.TestCase):
    def test_same_day_targets_do_not_enter_each_others_context(self):
        frame = pd.DataFrame({
            'user_id': [1, 1, 1, 1], 'item_id': [10, 20, 30, 40],
            'last_watch_dt': pd.to_datetime(['2021-01-01', '2021-01-02', '2021-01-02', '2021-01-03']),
            'watched_pct': [10., 20., np.nan, 40.],
        })
        data, _ = build_examples(frame, {10: 1, 20: 2, 30: 3, 40: 4}, 3)
        np.testing.assert_array_equal(data['input_ids'], [[1, 0, 0], [1, 0, 0], [1, 2, 3]])
        np.testing.assert_array_equal(data['targets'], [2, 3, 4])
        self.assertTrue(np.isnan(data['rewards'][1]))

    def test_unknown_items_are_not_padding_targets_or_context(self):
        frame = pd.DataFrame({
            'user_id': [1, 1, 1], 'item_id': [10, 999, 20],
            'last_watch_dt': pd.to_datetime(['2021-01-01', '2021-01-02', '2021-01-03']),
            'watched_pct': [10., 50., 100.],
        })
        data, counts = build_examples(frame, {10: 1, 20: 2}, 2)
        np.testing.assert_array_equal(data['input_ids'], [[1, 0]])
        self.assertEqual(counts['unknown_targets'], 1)


if __name__ == '__main__':
    unittest.main()
