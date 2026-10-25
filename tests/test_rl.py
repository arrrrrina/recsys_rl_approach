import unittest
import torch
from recsys_rl.rl import reinforce_loss, forward_kl, adjusted_rewards


class ReinforceTest(unittest.TestCase):
    def test_exact_expected_gradient_prefers_high_reward(self):
        # Enumerate all ordered pairs of independent actions and weight their
        # score gradients by sampling probabilities, checking unbiasedness.
        logits = torch.tensor([[0.2, -0.3]], requires_grad=True)
        lp = logits.log_softmax(-1)
        expected_gradient = torch.zeros_like(logits)
        rewards = torch.tensor([0., 1.])
        for a in range(2):
            for b in range(2):
                actions = torch.tensor([[a, b]])
                reward = rewards[actions]
                loss, _ = reinforce_loss(lp, actions, reward, lp.detach(), 0.)
                grad, = torch.autograd.grad(loss, logits, retain_graph=True)
                probability = (lp[0, a].exp() * lp[0, b].exp()).detach()
                expected_gradient += probability * grad
        true_loss = -(lp.exp() * rewards).sum()
        true_grad, = torch.autograd.grad(true_loss, logits)
        torch.testing.assert_close(expected_gradient, true_grad)

    def test_kl_identity_and_repeat_penalty(self):
        lp = torch.tensor([[0.4, 0.6]]).log()
        torch.testing.assert_close(forward_kl(lp, lp), torch.zeros(1))
        actual = adjusted_rewards(torch.tensor([[0.7, 0.8]]), torch.tensor([[2, 0]]),
                                  torch.tensor([[2, 3]]), 0.2)
        torch.testing.assert_close(actual, torch.tensor([[0.5, 0.8]]))


if __name__ == '__main__':
    unittest.main()
