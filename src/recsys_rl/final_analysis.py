"""Paired user uncertainty, synthetic histories, and observational return proxy."""
import numpy as np
import pandas as pd
import torch


def ranking_per_user(details, k=10):
    users, inverse = np.unique(details['user_ids'], return_inverse=True)
    counts = np.bincount(inverse)
    hits = details['predictions'][:, :k] == details['targets'][:, None]
    found = hits.any(axis=1)
    rank = hits.argmax(axis=1) + 1
    return {'user_ids': users,
            'ndcg@10': np.bincount(inverse, weights=found / np.log2(rank + 1)) / counts,
            'recall@10': np.bincount(inverse, weights=found.astype(float)) / counts}


def paired_bootstrap(baseline, candidate, samples=2000, seed=42):
    delta = np.asarray(candidate) - np.asarray(baseline)
    rng = np.random.default_rng(seed)
    means = []
    for start in range(0, samples, 100):
        indices = rng.integers(0, len(delta), size=(min(100, samples - start), len(delta)))
        means.extend(delta[indices].mean(axis=1))
    lower, upper = np.quantile(means, [0.025, 0.975])
    return {'difference': float(delta.mean()), 'ci95': [float(lower), float(upper)],
            'users': len(delta), 'bootstrap_samples': samples}


def append_items(input_ids, lengths, candidates):
    width = input_ids.shape[1]
    positions = torch.arange(width, device=input_ids.device)[None, :]
    shift = (lengths >= width).long()[:, None]
    result = input_ids.gather(1, (positions + shift).clamp(max=width - 1))
    new_lengths = (lengths + 1).clamp(max=width)
    result.scatter_(1, (new_lengths - 1)[:, None], candidates[:, None])
    return result, new_lengths


def rollout_indices(dataset, max_users=256, seed=42):
    # Use the first eligible test context per user, then sample users uniformly.
    _, indices = np.unique(dataset.arrays['user_ids'], return_index=True)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(indices, size=min(max_users, len(indices)), replace=False))


@torch.no_grad()
def rollout(model, reward_model, dataset, indices, device, mode='sample', seed=42, horizon=5):
    model.eval()
    reward_model.eval()
    generator = torch.Generator(device='cpu').manual_seed(seed)
    collected = {key: [] for key in ('reward', 'reward_frozen_history', 'repeat_initial',
                                    'repeat_generated', 'entropy')}
    generated = []
    for start in range(0, len(indices), 128):
        ix = indices[start:start + 128]
        initial = torch.as_tensor(dataset.arrays['input_ids'][ix], device=device)
        lengths = torch.as_tensor(dataset.arrays['lengths'][ix], device=device)
        inputs = initial.clone()
        frozen_history = reward_model.encode_history(initial, lengths)
        past = []
        batch = {key: [] for key in collected}
        for step in range(horizon):
            lp = model(inputs[:, :int(lengths.max().item())], lengths)[:, 1:].log_softmax(dim=-1)
            if mode == 'greedy':
                action = lp.argmax(dim=-1) + 1
            else:
                action = torch.multinomial(lp.exp().cpu(), 1, generator=generator).squeeze(1).to(device) + 1
            reward = reward_model(inputs, lengths, action)
            frozen_reward = reward_model.score_candidates(frozen_history, action)
            repeated = torch.zeros_like(action, dtype=torch.bool)
            if past:
                repeated = (action[:, None] == torch.stack(past, dim=1)).any(dim=1)
            for key, value in (
                ('reward', reward), ('reward_frozen_history', frozen_reward),
                ('repeat_initial', (action[:, None] == initial).any(dim=1).float()),
                ('repeat_generated', repeated.float()),
                ('entropy', -(lp.exp() * lp).sum(dim=-1)),
            ):
                batch[key].append(value.cpu().numpy())
            past.append(action)
            inputs, lengths = append_items(inputs, lengths, action)
        for key in collected:
            collected[key].append(np.stack(batch[key], axis=1))
        generated.append(torch.stack(past, dim=1).cpu().numpy())
    arrays = {key: np.concatenate(value) for key, value in collected.items()}
    tokens = np.concatenate(generated)
    discount = 0.9 ** np.arange(horizon)
    summary = {key: value.mean(axis=0).tolist() for key, value in arrays.items()}
    summary['users'] = len(indices)
    summary['mode'] = mode
    summary['discount_factor'] = 0.9
    summary['discounted_predicted_return'] = {
        str(h): float((arrays['reward'][:, :h] * discount[:h]).sum(axis=1).mean())
        for h in (1, 3, 5) if h <= horizon}
    summary['coverage_by_step'] = [int(len(np.unique(tokens[:, i]))) for i in range(horizon)]
    summary['mean_unique_items_in_chain'] = float(np.mean([len(np.unique(row)) for row in tokens]))
    return summary, tokens


def return_proxy(raw_path, selected_users, train_cutoff, validation_cutoff):
    frame = pd.read_csv(raw_path, usecols=['user_id', 'last_watch_dt', 'watched_pct'])
    dates = pd.to_datetime(frame.last_watch_dt, format='%Y-%m-%d', errors='coerce')
    observation_end = dates.max()
    frame['date'] = dates
    frame = frame[frame.user_id.isin(selected_users) & dates.notna()]
    days = frame.groupby(['user_id', 'date'], as_index=False).watched_pct.mean()
    days = days.sort_values(['user_id', 'date'])
    days['next_date'] = days.groupby('user_id').date.shift(-1)
    days['return_7d'] = (days.next_date - days.date).dt.days.between(1, 7).astype(float)
    output = {}
    for name, low, high, end in (
        ('train', pd.Timestamp.min, pd.Timestamp(train_cutoff), pd.Timestamp(train_cutoff) - pd.Timedelta(days=1)),
        ('validation', pd.Timestamp(train_cutoff), pd.Timestamp(validation_cutoff), pd.Timestamp(validation_cutoff) - pd.Timedelta(days=1)),
        ('test', pd.Timestamp(validation_cutoff), pd.Timestamp.max, observation_end),
    ):
        eligible = days[(days.date >= low) & (days.date < high) &
                        (days.date + pd.Timedelta(days=7) <= end)]
        output[name] = {'complete_window_user_days': len(eligible),
                        'return_7d_rate': float(eligible.return_7d.mean()) if len(eligible) else None}
    output['limitations'] = ('Descriptive next-observed-day proxy, not a reward model or causal retention estimate. '
                            'KION stores aggregated user-item rows and last_watch_dt, so missing daily events '
                            'may distort returns. Complete seven-day windows only; no crossing split boundaries. '
                            'A return predictor would need only decision-time history; online validation is required '
                            'to establish that a recommendation changes retention.')
    return output
