import unittest
import numpy as np
import torch
from recsys_rl.reward import RewardModel, fit_item_prior, regression_metrics


class RewardModelTest(unittest.TestCase):
    def test_missing_feedback_and_unseen_item_prior(self):
        prior, counts, mean = fit_item_prior(np.array([1, 1, 2]), np.array([1., np.nan, 0.]), 4, 2)
        self.assertAlmostEqual(mean, 0.5)
        self.assertEqual(counts[1], 1)
        self.assertAlmostEqual(prior[3], 0.5)

    def test_candidate_scoring_is_batch_consistent_and_padding_invariant(self):
        torch.manual_seed(42)
        model = RewardModel(torch.randn(8, 4), np.full(8, 0.6), dropout=0).eval()
        # A nonzero residual makes the test sensitive to the encoded context.
        torch.nn.init.normal_(model.residual[-1].weight, std=0.1)
        inputs, lengths = torch.tensor([[1, 2, 0], [3, 0, 0]]), torch.tensor([2, 1])
        candidates = torch.tensor([[4, 5], [5, 6]])
        together = model(inputs, lengths, candidates)
        for i in range(2):
            torch.testing.assert_close(together[:, i], model(inputs, lengths, candidates[:, i]))
        torch.testing.assert_close(model(inputs, lengths, candidates[:, 0]),
                                   model(inputs[:, :2], lengths, candidates[:, 0]))
        loss = together.mean()
        loss.backward()
        self.assertTrue(torch.isfinite(model.embedding.weight.grad).all())

    def test_regression_perfect_prediction(self):
        metrics = regression_metrics(np.array([0., 1.]), np.array([0., 1.]))
        self.assertEqual(metrics['rmse'], 0.)
        self.assertEqual(metrics['r2'], 1.)


if __name__ == '__main__':
    unittest.main()
