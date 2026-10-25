"""Bounded reward regression with a train-only smoothed item prior."""
import numpy as np
import torch
from torch import nn


def fit_item_prior(targets, rewards, vocab_size, smoothing=20.0):
    valid = np.isfinite(rewards)
    if not valid.any():
        raise ValueError('No observed rewards in train')
    targets, rewards = targets[valid], rewards[valid]
    global_mean = float(rewards.mean())
    counts = np.bincount(targets, minlength=vocab_size)
    sums = np.bincount(targets, weights=rewards, minlength=vocab_size)
    means = (sums + smoothing * global_mean) / (counts + smoothing)
    return means.astype(np.float32), counts, global_mean


class RewardModel(nn.Module):
    """History + candidate -> expected watched_pct / 100.

    The prior is fitted only on observed train targets. The neural residual
    captures history/candidate interactions. Missing feedback is never a zero.
    Candidate IDs can be [batch] or [batch, candidates], for later RL/reranking.
    """
    def __init__(self, item_embeddings, item_prior, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(
            item_embeddings.detach().float().clone(), freeze=False, padding_idx=0)
        dim = item_embeddings.shape[1]
        self.norm = nn.LayerNorm(dim)
        self.residual = nn.Sequential(
            nn.Linear(4 * dim, 128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, 1),
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)
        prior = torch.as_tensor(item_prior, dtype=torch.float32).clamp(1e-4, 1 - 1e-4)
        self.register_buffer('prior_logits', torch.logit(prior))

    def encode_history(self, input_ids, lengths):
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        valid = positions[None, :] < lengths[:, None]
        # More recent observed objects receive larger pooling weights.
        ages = (lengths[:, None] - 1 - positions[None, :]).clamp(min=0).float()
        weights = (0.95 ** ages) * valid.float()
        history = (self.embedding(input_ids) * weights[:, :, None]).sum(dim=1)
        return self.norm(history / weights.sum(dim=1, keepdim=True).clamp(min=1))

    def score_candidates(self, history, candidates):
        candidate = self.norm(self.embedding(candidates))
        if candidates.ndim == 2:
            history = history[:, None, :].expand_as(candidate)
        features = torch.cat([history, candidate, history * candidate,
                              (history - candidate).abs()], dim=-1)
        return torch.sigmoid(self.prior_logits[candidates] + self.residual(features).squeeze(-1))

    def forward(self, input_ids, lengths, candidates):
        return self.score_candidates(self.encode_history(input_ids, lengths), candidates)


def regression_metrics(target, prediction):
    target, prediction = np.asarray(target), np.asarray(prediction)
    error = prediction - target
    denominator = ((target - target.mean()) ** 2).sum()
    correlation = None
    if np.std(target) > 0 and np.std(prediction) > 1e-8:
        correlation = float(np.corrcoef(target, prediction)[0, 1])
    return {
        'mae': float(np.abs(error).mean()),
        'rmse': float(np.sqrt((error ** 2).mean())),
        'r2': float(1 - (error ** 2).sum() / denominator) if denominator > 0 else None,
        'pearson': correlation,
        'mean_prediction': float(prediction.mean()),
        'mean_observed_reward': float(target.mean()),
        'examples': len(target),
    }


def calibration_bins(target, prediction):
    bins = []
    for low in np.linspace(0, 0.9, 10):
        high = low + 0.1
        mask = (prediction >= low) & ((prediction < high) if high < 0.999 else (prediction <= 1))
        if mask.any():
            bins.append({'low': float(low), 'high': float(high), 'count': int(mask.sum()),
                         'prediction': float(prediction[mask].mean()),
                         'observed': float(target[mask].mean())})
    return bins
