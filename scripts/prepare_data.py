"""Prepare data: python3 scripts/prepare_data.py."""
from _bootstrap import bootstrap
bootstrap()
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import platform

import numpy as np
import pandas as pd
import torch

from recsys_rl.data import build_examples


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=Path('../RecSys_feature_engineering/data'))
    parser.add_argument('--config', type=Path, default=Path('configs/small.json'))
    parser.add_argument('--run-dir', type=Path, default=Path('.'))
    parser.add_argument('--freeze-eval-from', type=Path, default=None,
                        help='Keep this run\'s validation/test, vocabulary and original train users')
    args = parser.parse_args(argv)
    cfg = json.loads(args.config.read_text())
    raw = Path('data/raw')
    out = args.run_dir / 'artifacts/prepared'
    reports = args.run_dir / 'reports'
    reports.mkdir(parents=True, exist_ok=True)
    raw.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    provenance = {}
    for name in ('interactions.csv', 'items.csv'):
        source = args.source / name
        destination = raw / name
        if not destination.exists():
            shutil.copy2(source, destination)
        source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        digest = hashlib.sha256(destination.read_bytes()).hexdigest()
        if digest != source_hash:
            raise ValueError(f'Local data differs from source: {name}')
        provenance[name] = {'source': str(source.resolve()), 'sha256': digest}
    df = pd.read_csv(raw / 'interactions.csv')
    source_rows = len(df)
    dates = pd.to_datetime(df.last_watch_dt, format='%Y-%m-%d', errors='coerce')
    invalid = df.loc[dates.isna()].to_dict('records')
    # Preserve the malformed records for inspection; never silently infer dates.
    df.loc[dates.isna()].to_csv(reports / 'invalid_interactions.csv', index=False)
    df = df.loc[dates.notna()].copy()
    df['last_watch_dt'] = dates.loc[dates.notna()]
    reward = df.watched_pct
    if ((reward.dropna() < 0) | (reward.dropna() > 100)).any():
        raise ValueError('watched_pct is outside [0,100]')
    train = df[df.last_watch_dt < cfg['train_cutoff']]
    item_ids = sorted(int(x) for x in train.item_id.unique())
    vocabulary = {item: i + 1 for i, item in enumerate(item_ids)}
    day_counts = train.groupby('user_id').last_watch_dt.nunique()
    eligible = day_counts[day_counts >= cfg['min_train_days']].index.to_numpy()
    rng = np.random.default_rng(cfg['seed'])
    frozen = None
    frozen_hashes = {}
    if args.freeze_eval_from is not None:
        frozen = args.freeze_eval_from / 'artifacts/prepared'
        if frozen.resolve() == out.resolve():
            raise ValueError('Frozen source must differ from output')
        old_audit = json.loads((args.freeze_eval_from / 'reports/data_audit.json').read_text())
        if provenance != old_audit['provenance']:
            raise ValueError('Frozen evaluation comes from different raw data')
        for key in ('min_train_days', 'max_length', 'train_cutoff', 'validation_cutoff'):
            if cfg[key] != old_audit['config'][key]:
                raise ValueError(f'Frozen evaluation protocol mismatch: {key}')
        if item_ids != json.loads((frozen / 'vocabulary.json').read_text()):
            raise ValueError('Frozen item vocabulary mismatch')
        original_users = np.load(frozen / 'selected_users.npy')
        if not np.isin(original_users, eligible).all():
            raise ValueError('Original users no longer eligible')
        if cfg['max_users'] < len(original_users) or cfg['max_users'] > len(eligible):
            raise ValueError('Requested size must include original users and fit eligible pool')
        remaining = np.setdiff1d(eligible, original_users)
        extra = rng.choice(remaining, size=cfg['max_users'] - len(original_users), replace=False)
        users = np.sort(np.concatenate([original_users, extra]))
    else:
        users = np.sort(rng.choice(eligible, size=min(cfg['max_users'], len(eligible)), replace=False))
    selected = df[df.user_id.isin(users)]
    arrays, counters = build_examples(selected, vocabulary, cfg['max_length'])
    first_boundary = np.datetime64(cfg['train_cutoff'], 'D').astype('int64')
    second_boundary = np.datetime64(cfg['validation_cutoff'], 'D').astype('int64')
    days = arrays['target_days']
    masks = {
        'train': days < first_boundary,
        'validation': (days >= first_boundary) & (days < second_boundary),
        'test': days >= second_boundary,
    }
    statistics = {}
    for split, mask in masks.items():
        if frozen is not None and split in ('validation', 'test'):
            source_file = frozen / f'{split}.npz'
            shutil.copy2(source_file, out / f'{split}.npz')
            with np.load(source_file) as archive:
                part = {key: archive[key] for key in archive.files}
            source_hash = hashlib.sha256(source_file.read_bytes()).hexdigest()
            if source_hash != hashlib.sha256((out / f'{split}.npz').read_bytes()).hexdigest():
                raise ValueError(f'Frozen split copy mismatch: {split}')
            frozen_hashes[split] = source_hash
        else:
            part = {key: value[mask] for key, value in arrays.items()}
            np.savez_compressed(out / f'{split}.npz', **part)
        statistics[split] = {
            'examples': len(part['targets']),
            'users': int(len(np.unique(part['user_ids']))),
            'reward_examples': int(np.isfinite(part['rewards']).sum()),
            'mean_history_length': float(part['lengths'].mean()) if len(part['targets']) else None,
        }
    (out / 'vocabulary.json').write_text(json.dumps(item_ids))
    np.save(out / 'selected_users.npy', users)
    counts = train.item_id.value_counts()
    np.save(out / 'train_popularity.npy', np.array([0] + [int(counts.get(x, 0)) for x in item_ids]))
    report = {
        'config': cfg, 'provenance': provenance,
        'runtime': {'python': platform.python_version(), 'torch': torch.__version__,
                    'cuda': torch.cuda.is_available(), 'mps': torch.backends.mps.is_available()},
        'source_rows': source_rows, 'valid_rows': len(df),
        'invalid_date_rows': len(invalid),
        'date_range': [str(df.last_watch_dt.min()), str(df.last_watch_dt.max())],
        'unique_users': int(df.user_id.nunique()),
        'unique_items': int(df.item_id.nunique()),
        'duplicate_user_item_pairs': int(df.duplicated(['user_id', 'item_id']).sum()),
        'same_user_day_extra_events': int(df.duplicated(['user_id', 'last_watch_dt']).sum()),
        'missing_reward_rows': int(df.watched_pct.isna().sum()),
        'full_split_rows': {
            'train': len(train),
            'validation': int(((df.last_watch_dt >= cfg['train_cutoff']) &
                               (df.last_watch_dt < cfg['validation_cutoff'])).sum()),
            'test': int((df.last_watch_dt >= cfg['validation_cutoff']).sum()),
        },
        'eligible_users': len(eligible), 'selected_users': len(users),
        'vocabulary_items': len(vocabulary), 'excluded_selected_events': counters,
        'prepared': statistics,
        'frozen_evaluation_source': str(args.freeze_eval_from.resolve()) if frozen is not None else None,
        'frozen_split_sha256': frozen_hashes,
        'original_train_users_included': len(original_users) if frozen is not None else None,
        'protocol': 'Single next-event target; strictly earlier-day history; train-only vocabulary and user selection. All held-out events evaluated with previously observed history; unknown targets excluded and reported. Same-day history ties sorted by item_id as an explicit approximation. Missing rewards excluded only from reward-model fitting.',
    }
    (reports / 'data_audit.json').write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
