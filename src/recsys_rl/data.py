"""Build next-item examples with strictly earlier-day contexts.

KION has day-resolution dates: item_id is a deterministic ordering inside a
past day, not a recovered temporal order. No same-day event enters the context
of its target. Training must use the final context position only, rather than
an autoregressive loss across artificially ordered events from the same day.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def build_examples(frame, item_to_token, max_length=64):
    """Return padded contexts and single next-event labels; PAD=0."""
    examples = {k: [] for k in (
        'input_ids', 'lengths', 'targets', 'rewards', 'user_ids',
        'target_days', 'history_last_days',
    )}
    counters = {'unknown_targets': 0, 'empty_history': 0}
    ordered = frame.sort_values(['user_id', 'last_watch_dt', 'item_id'])
    for user, group in ordered.groupby('user_id', sort=False):
        days = group.last_watch_dt.to_numpy(dtype='datetime64[D]').astype('int64')
        tokens = group.item_id.map(item_to_token).fillna(0).to_numpy(dtype=np.int64)
        rewards = group.watched_pct.to_numpy(dtype=np.float32) / 100.0
        for i, (day, target) in enumerate(zip(days, tokens)):
            if target == 0:
                counters['unknown_targets'] += 1
                continue
            end = np.searchsorted(days, day, side='left')
            known = np.flatnonzero(tokens[:end] > 0)[-max_length:]
            if len(known) == 0:
                counters['empty_history'] += 1
                continue
            context = np.zeros(max_length, dtype=np.int64)
            context[:len(known)] = tokens[known]
            examples['input_ids'].append(context)
            examples['lengths'].append(len(known))
            examples['targets'].append(target)
            examples['rewards'].append(rewards[i])
            examples['user_ids'].append(user)
            examples['target_days'].append(day)
            examples['history_last_days'].append(days[known[-1]])
    arrays = {}
    for key, values in examples.items():
        dtype = np.float32 if key == 'rewards' else np.int64
        arrays[key] = np.asarray(values, dtype=dtype)
    arrays['input_ids'] = arrays['input_ids'].reshape(-1, max_length)
    assert np.all(arrays['history_last_days'] < arrays['target_days'])
    assert np.all(arrays['targets'] > 0)
    return arrays, counters
