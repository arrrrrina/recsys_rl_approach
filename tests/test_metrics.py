import unittest
import numpy as np
from recsys_rl.evaluation import next_item_metrics


class RankingMetricsTest(unittest.TestCase):
    def test_macro_users_and_missed_target(self):
        # User 1 has two events, user 2 one; each user has equal final weight.
        metrics = next_item_metrics(
            [[4, 2, 3], [5, 3, 2], [8, 9, 7]], [2, 99, 8], [1, 1, 2], ks=(1, 3))
        self.assertAlmostEqual(metrics['recall@1'], 0.5)
        self.assertAlmostEqual(metrics['recall@3'], 0.75)
        self.assertAlmostEqual(metrics['ndcg@3'], (1 / np.log2(3) / 2 + 1) / 2)


if __name__ == '__main__':
    unittest.main()
