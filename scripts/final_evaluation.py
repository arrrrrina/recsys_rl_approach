"""Freeze validation selection, evaluate test once, analyze five-step generation."""
from _bootstrap import bootstrap
bootstrap()
from pathlib import Path
import argparse
import hashlib
import json
import platform

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from recsys_rl.checkpoints import load_policy, load_reward
from recsys_rl.evaluation import PreparedDataset, evaluate, evaluate_popularity, move_context
from recsys_rl.policy_analysis import analyze_policy
from recsys_rl.reward import regression_metrics, fit_item_prior
from recsys_rl.final_analysis import ranking_per_user, paired_bootstrap, rollout_indices, rollout, return_proxy
from recsys_rl.runtime import resolve_device, seed_everything


def plot_results(folder, result, validation):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for ax, split in zip(axes, ('validation', 'test')):
        values = {'baseline': validation['baseline_validation'],
                  **{name: value['selected'] for name, value in validation['variants'].items()}} if split == 'validation' else result['test']
        for name, row in values.items():
            if 'reward@10' not in row:
                continue
            ax.scatter(row['ndcg@10'], row['reward@10'])
            ax.annotate(name, (row['ndcg@10'], row['reward@10']), fontsize=7, xytext=(3, 3), textcoords='offset points')
        ax.set(xlabel='NDCG@10', ylabel='Predicted reward@10', title=split.title())
        ax.grid(alpha=0.2)
    fig.savefig(folder / 'tradeoff.png', dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for name, modes in result['rollout'].items():
        values = modes['sample']
        axes[0].plot(range(1, 6), values['reward'], marker='o', label=name)
        axes[1].plot(range(1, 6), values['repeat_generated'], marker='o', label=name)
    axes[0].set(xlabel='Step', ylabel='Predicted reward', title='Sampling: synthetic histories')
    axes[1].set(xlabel='Step', ylabel='Repeat of generated item', title='Sampling: repeats')
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7)
    fig.savefig(folder / 'rollout.png', dpi=180)
    plt.close(fig)


@torch.no_grad()
def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, default=Path('experiments/users50k'))
    parser.add_argument('--device', choices=['mps', 'cpu', 'cuda'], default='mps')
    parser.add_argument('--bootstrap-samples', type=int, default=2000)
    parser.add_argument('--rollout-users', type=int, default=256)
    args = parser.parse_args(argv)
    if args.bootstrap_samples < 1 or args.rollout_users < 1:
        raise ValueError('bootstrap-samples and rollout-users must be positive')
    device = resolve_device(args.device)
    seed_everything(42)
    root = args.run_dir
    reports = root / 'reports'
    prepared = root / 'artifacts/prepared'
    validation = json.loads((reports / 'rl.json').read_text())
    audit = json.loads((reports / 'data_audit.json').read_text())
    paths = {'baseline': root / 'artifacts/baseline/best.pt',
             **{name: root / 'artifacts/rl' / name / 'best.pt' for name in validation['variants']}}
    selection = {'main_rl_variant': validation['recommended_variant'],
                 'checkpoint_sha256': {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()},
                 'rule': validation['selection_rule'],
                 'test_sha256': hashlib.sha256((prepared / 'test.npz').read_bytes()).hexdigest(),
                 'reward_checkpoint_sha256': hashlib.sha256((root / 'artifacts/reward/best.pt').read_bytes()).hexdigest()}
    selection_path = reports / 'selection_before_test.json'
    if selection_path.exists() and json.loads(selection_path.read_text()) != selection:
        raise ValueError('Model selection changed after test registration; use a separate experiment')
    selection_path.write_text(json.dumps(selection, indent=2))
    test = PreparedDataset(prepared / 'test.npz')
    train = PreparedDataset(prepared / 'train.npz')
    popularity = np.load(prepared / 'train_popularity.npy')
    _, item_counts, _ = fit_item_prior(train.arrays['targets'], train.arrays['rewards'], len(popularity))
    reference, baseline_cp = load_policy(paths['baseline'], device)
    reference.eval().requires_grad_(False)
    reward_model, reward_cp = load_reward(root / 'artifacts/reward/best.pt', device)
    if baseline_cp['data_provenance'] != reward_cp['data_provenance'] or baseline_cp['data_provenance'] != audit['provenance']:
        raise ValueError('Data provenance differs')
    result = {'selection': selection, 'test': {'popularity': evaluate_popularity(test, popularity)},
              'bootstrap': {}, 'rollout': {}, 'runtime': {'python': platform.python_version(),
              'torch': str(torch.__version__), 'numpy': str(np.__version__), 'device': str(device)}}
    indices = rollout_indices(test, args.rollout_users, seed=42)
    per_user = {}
    predictions_archive = {}
    examples = {}
    vocabulary = json.loads((prepared / 'vocabulary.json').read_text())
    items = pd.read_csv('data/raw/items.csv', usecols=['item_id', 'title']).set_index('item_id').title.to_dict()
    for name, path in paths.items():
        model, _ = load_policy(path, device)
        metrics, details = evaluate(model, test, device, 256, popularity, f'Test {name}', return_details=True)
        diagnostics, reward_users = analyze_policy(model, reference, reward_model, test, device,
            batch_size=256, samples=8, item_counts=item_counts, return_per_user=True)
        result['test'][name] = {**metrics, **diagnostics}
        user_metrics = ranking_per_user(details)
        np.testing.assert_array_equal(user_metrics['user_ids'], reward_users['user_ids'])
        user_metrics['reward@10'] = reward_users['reward@10']
        per_user[name] = user_metrics
        predictions_archive[name] = details['predictions']
        if name != 'baseline':
            result['bootstrap'][name] = {key: paired_bootstrap(per_user['baseline'][key], user_metrics[key],
                samples=args.bootstrap_samples) for key in ('ndcg@10', 'recall@10', 'reward@10')}
        result['rollout'][name] = {}
        examples[name] = {}
        for mode in ('greedy', 'sample'):
            summary, tokens = rollout(model, reward_model, test, indices, device, mode=mode)
            result['rollout'][name][mode] = summary
            examples[name][mode] = []
            for i in range(min(5, len(indices))):
                item_ids = [vocabulary[token - 1] for token in tokens[i]]
                history_tokens = test.arrays['input_ids'][indices[i]][:test.arrays['lengths'][indices[i]]]
                history_ids = [vocabulary[token - 1] for token in history_tokens[-10:]]
                examples[name][mode].append({'user_id': int(test.arrays['user_ids'][indices[i]]),
                    'history_last10_titles': [str(items.get(item, item)) for item in history_ids],
                    'generated_item_ids': item_ids, 'generated_titles': [str(items.get(item, item)) for item in item_ids]})
        print(name, json.dumps(result['test'][name]), flush=True)
        (reports / 'final_evaluation.json').write_text(json.dumps(result, indent=2))
        del model
        if device.type == 'mps':
            torch.mps.empty_cache()
    # Reward model generalization on observed test outcomes, not generated actions.
    observed_predictions = []
    for batch in DataLoader(test, batch_size=512, num_workers=0):
        inputs, lengths = move_context(batch, device)
        observed_predictions.append(reward_model(inputs, lengths, batch['targets'].to(device)).cpu().numpy())
    observed_predictions = np.concatenate(observed_predictions)
    valid = np.isfinite(test.arrays['rewards'])
    y = test.arrays['rewards'][valid]
    prior = reward_cp['item_prior'].numpy()
    result['reward_test'] = {
        'constant_train_mean': regression_metrics(y, np.full_like(y, reward_cp['global_train_mean'])),
        'smoothed_item_mean': regression_metrics(y, prior[test.arrays['targets'][valid]]),
        'reward_model': regression_metrics(y, observed_predictions[valid]),
    }
    result['return_proxy'] = return_proxy('data/raw/interactions.csv', np.load(prepared / 'selected_users.npy'),
                                          audit['config']['train_cutoff'], audit['config']['validation_cutoff'])
    predictions_archive.update({'targets': test.arrays['targets'], 'user_ids': test.arrays['user_ids']})
    np.savez_compressed(reports / 'test_predictions.npz', **predictions_archive)
    (reports / 'rollout_examples.json').write_text(json.dumps(examples, indent=2, ensure_ascii=False))
    (reports / 'final_evaluation.json').write_text(json.dumps(result, indent=2))
    plot_results(reports, result, validation)
    print(f'\nSaved {reports / "final_evaluation.json"} and plots', flush=True)
    print('Test comparison:')
    for name, row in result['test'].items():
        print(f'{name:20} NDCG@10={row["ndcg@10"]:.5f} reward@10={row.get("reward@10", float("nan")):.4f}')


if __name__ == '__main__':
    main()
