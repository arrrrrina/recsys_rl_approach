"""One-step REINFORCE with a leave-one-out baseline and exact forward KL."""
import torch


def adjusted_rewards(raw_reward, input_ids, candidates, repeat_penalty=0.0):
    # The penalty concerns the observable truncated context, not lifetime history.
    repeats = (candidates[:, :, None] == input_ids[:, None, :]).any(dim=-1)
    return raw_reward - repeat_penalty * repeats.float()


def forward_kl(log_probs, reference_log_probs):
    return (log_probs.exp() * (log_probs - reference_log_probs)).sum(dim=-1)


def reinforce_loss(log_probs, actions, rewards, reference_log_probs, beta):
    """actions are zero-based real-item indices: PAD already removed.

    Samples must be independent draws WITH replacement. The mean of the other
    samples is independent of the current action, giving an unbiased baseline.
    Do not normalize advantages across the group: that changes the objective.
    """
    if rewards.shape[1] < 2:
        raise ValueError('Leave-one-out baseline requires at least two samples')
    other_mean = (rewards.sum(dim=1, keepdim=True) - rewards) / (rewards.shape[1] - 1)
    advantage = (rewards - other_mean).detach()
    policy_loss = -(log_probs.gather(1, actions) * advantage).mean()
    kl = forward_kl(log_probs, reference_log_probs).mean()
    return policy_loss + beta * kl, kl
