"""Diagnostics on recommendations, separate from observed-response accuracy."""
import numpy as np
import torch
from torch.utils.data import DataLoader

from recsys_rl.evaluation import move_context
from recsys_rl.rl import adjusted_rewards, forward_kl


@torch.no_grad()
def analyze_policy(model, reference, reward_model, dataset, device, batch_size=256,
                   beta=0.0, repeat_penalty=0.0, samples=4, seed=142, item_counts=None,
                   return_per_user=False):
    model.eval()
    reference.eval()
    reward_model.eval()
    # Dedicated CPU sampling generator: evaluation does not reset MPS training RNG.
    generator = torch.Generator(device='cpu').manual_seed(seed)
    records = {key: [] for key in ('reward@10', 'adjusted_reward@10', 'sampled_reward',
                                  'sampled_adjusted_reward', 'kl_to_reference')}
    if item_counts is not None:
        records['low_reward_support_share@10'] = []
    users = []
    for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0):
        inputs, lengths = move_context(batch, device)
        log_probs = model(inputs, lengths)[:, 1:].log_softmax(dim=-1)
        reference_log_probs = reference(inputs, lengths)[:, 1:].log_softmax(dim=-1)
        top = log_probs.topk(min(10, log_probs.shape[1]), dim=-1).indices + 1
        history = reward_model.encode_history(inputs, lengths)
        top_reward = reward_model.score_candidates(history, top)
        adjusted = adjusted_rewards(top_reward, inputs, top, repeat_penalty)
        actions = torch.multinomial(log_probs.exp().cpu(), samples, replacement=True,
                                    generator=generator).to(device) + 1
        sample_reward = reward_model.score_candidates(history, actions)
        sample_adjusted = adjusted_rewards(sample_reward, inputs, actions, repeat_penalty)
        for key, value in (
            ('reward@10', top_reward.mean(dim=1)),
            ('adjusted_reward@10', adjusted.mean(dim=1)),
            ('sampled_reward', sample_reward.mean(dim=1)),
            ('sampled_adjusted_reward', sample_adjusted.mean(dim=1)),
            ('kl_to_reference', forward_kl(log_probs, reference_log_probs)),
        ):
            records[key].append(value.cpu().numpy())
        if item_counts is not None:
            records['low_reward_support_share@10'].append((item_counts[top.cpu().numpy()] < 5).mean(axis=1))
        users.append(batch['user_ids'].numpy())
    users = np.concatenate(users)
    _, inverse = np.unique(users, return_inverse=True)
    counts = np.bincount(inverse)
    result = {}
    per_user = {'user_ids': np.unique(users)}
    for key, parts in records.items():
        values = np.concatenate(parts)
        per_user[key] = np.bincount(inverse, weights=values) / counts
        result[key] = float(per_user[key].mean())
    result['regularized_sampled_objective'] = result['sampled_adjusted_reward'] - beta * result['kl_to_reference']
    return (result, per_user) if return_per_user else result
