import unittest
import numpy as np
import torch
from recsys_rl.final_analysis import append_items, paired_bootstrap


class FinalAnalysisTest(unittest.TestCase):
    def test_append_grows_or_truncates_without_mixing_users(self):
        ids, lengths = append_items(torch.tensor([[1, 2, 0], [4, 5, 6]]),
                                    torch.tensor([2, 3]), torch.tensor([3, 7]))
        torch.testing.assert_close(ids, torch.tensor([[1, 2, 3], [5, 6, 7]]))
        torch.testing.assert_close(lengths, torch.tensor([3, 3]))

    def test_paired_bootstrap_identity_and_constant_effect(self):
        base = np.array([1., 2., 3.])
        result = paired_bootstrap(base, base + 0.5, samples=200)
        self.assertAlmostEqual(result['difference'], 0.5)
        np.testing.assert_allclose(result['ci95'], [0.5, 0.5])


if __name__ == '__main__':
    unittest.main()
