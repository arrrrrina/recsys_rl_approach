"""Compare frozen baseline, continued SFT, and four one-step RL variants."""
from _bootstrap import bootstrap
bootstrap()
from pathlib import Path
import argparse
import json
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from recsys_rl.checkpoints import load_policy, load_reward
from recsys_rl.evaluation import PreparedDataset, evaluate, move_context
from recsys_rl.policy_analysis import analyze_policy
from recsys_rl.reward import fit_item_prior
from recsys_rl.rl import adjusted_rewards, reinforce_loss
from recsys_rl.runtime import resolve_device, seed_everything


def save_checkpoint(path, model, baseline, variant, step, metrics, run_config):
    torch.save({
        'state_dict': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        'config': baseline['config'], 'vocab_size': baseline['vocab_size'],
        'data_provenance': baseline['data_provenance'],
        'variant': variant, 'step': step, 'validation': metrics, 'rl_config': run_config,
    }, path)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=Path('configs/rl.json'))
    parser.add_argument('--run-dir', type=Path, default=Path('.'))
    parser.add_argument('--device', choices=['mps', 'cpu', 'cuda'], default=None)
    parser.add_argument('--steps', type=int, default=None)
    args = parser.parse_args(argv)
    cfg = json.loads(args.config.read_text())
    if args.device is not None:
        cfg['device'] = args.device
    if args.steps is not None:
        cfg['steps'] = args.steps
    if cfg['steps'] < 1:
        raise ValueError('steps must be positive')
    device = resolve_device(cfg['device'])
    seed_everything(cfg['seed'])
    prepared = args.run_dir / 'artifacts/prepared'
    output = args.run_dir / 'artifacts/rl'
    reports = args.run_dir / 'reports'
    reports.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    train = PreparedDataset(prepared / 'train.npz')
    validation = PreparedDataset(prepared / 'validation.npz')
    popularity = np.load(prepared / 'train_popularity.npy')
    reference, baseline = load_policy(args.run_dir / 'artifacts/baseline/best.pt', device)
    reference.eval().requires_grad_(False)
    reward_model, reward_checkpoint = load_reward(args.run_dir / 'artifacts/reward/best.pt', device)
    audit = json.loads((reports / 'data_audit.json').read_text())
    if not (baseline['data_provenance'] == reward_checkpoint['data_provenance'] == audit['provenance']):
        raise ValueError('Checkpoint data provenance mismatch')
    for key in ('seed', 'max_users', 'min_train_days', 'max_length', 'train_cutoff', 'validation_cutoff'):
        if baseline['config'][key] != audit['config'][key]:
            raise ValueError(f'Prepared data mismatch: {key}')
    _, item_counts, _ = fit_item_prior(train.arrays['targets'], train.arrays['rewards'], len(popularity))

    def assess(policy, variant):
        ranking = evaluate(policy, validation, device, cfg['eval_batch_size'], popularity)
        reward = analyze_policy(policy, reference, reward_model, validation, device,
                                cfg['eval_batch_size'], variant['beta'], variant['repeat_penalty'],
                                cfg['eval_samples'], cfg['seed'] + 100, item_counts)
        return {**ranking, **reward}

    baseline_metrics = assess(reference, {'beta': 0.0, 'repeat_penalty': 0.0})
    floor = baseline_metrics['ndcg@10'] * (1 - cfg['max_relative_ndcg_drop'])
    report = {'config': cfg, 'baseline_validation': baseline_metrics,
              'ndcg_floor': floor, 'variants': {},
              'selection_rule': 'Within each variant maximize validation regularized sampled objective among checkpoints with NDCG@10 >= floor. If none qualify, retain highest-NDCG checkpoint and explicitly flag constraint failure. Cross-variant choice uses raw reward@10 subject to the same NDCG floor. Test is untouched.'}
    print('Baseline validation:', json.dumps(baseline_metrics), flush=True)
    variants = list(cfg['variants'])
    if cfg['run_sft_control']:
        variants.insert(0, {'name': 'sft_control', 'beta': 0.0, 'repeat_penalty': 0.0, 'sft': True})
    run_start = time.perf_counter()
    for variant in variants:
        seed_everything(cfg['seed'])
        model, _ = load_policy(args.run_dir / 'artifacts/baseline/best.pt', device)
        # Dropout off in both policies: KL compares their actual recommendation distributions.
        # eval() does not disable gradients; all policy parameters remain trainable.
        model.eval()
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'],
                                     weight_decay=cfg['weight_decay'], foreach=False)
        loader = DataLoader(train, batch_size=cfg['batch_size'], shuffle=True, num_workers=0)
        iterator = iter(loader)
        variant_dir = output / variant['name']
        variant_dir.mkdir(parents=True, exist_ok=True)
        evaluations = []
        train_logs = []
        best_key = None
        progress = tqdm(range(1, cfg['steps'] + 1), desc=variant['name'])
        for step in progress:
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            inputs, lengths = move_context(batch, device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs, lengths)
            if variant.get('sft', False):
                loss = torch.nn.functional.cross_entropy(logits, batch['targets'].to(device))
                kl_value = 0.0
                raw_mean = None
                adjusted_mean = None
            else:
                log_probs = logits[:, 1:].log_softmax(dim=-1)
                with torch.no_grad():
                    reference_log_probs = reference(inputs, lengths)[:, 1:].log_softmax(dim=-1)
                    actions = torch.multinomial(log_probs.detach().exp(), cfg['group_size'], replacement=True)
                    candidates = actions + 1
                    raw = reward_model(inputs, lengths, candidates)
                    adjusted = adjusted_rewards(raw, inputs, candidates, variant['repeat_penalty'])
                loss, kl = reinforce_loss(log_probs, actions, adjusted, reference_log_probs, variant['beta'])
                kl_value = kl.item()
                raw_mean, adjusted_mean = raw.mean().item(), adjusted.mean().item()
            if not torch.isfinite(loss).item():
                raise RuntimeError(f'Non-finite loss: {variant["name"]}, step {step}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['gradient_clip'], foreach=False)
            optimizer.step()
            train_logs.append({'step': step, 'loss': loss.item(), 'kl': kl_value,
                               'sampled_reward': raw_mean, 'adjusted_reward': adjusted_mean})
            progress.set_postfix(loss=f'{loss.item():.4f}', kl=f'{kl_value:.4f}')
            if step % cfg['eval_every'] == 0 or step == cfg['steps']:
                metrics = assess(model, variant)
                feasible = metrics['ndcg@10'] >= floor
                row = {'step': step, 'meets_ndcg_constraint': feasible, **metrics}
                evaluations.append(row)
                save_checkpoint(variant_dir / f'step_{step:04d}.pt', model, baseline, variant, step, metrics, cfg)
                # Lexicographic choice keeps a valid checkpoint over any invalid one.
                key = (int(feasible), metrics['regularized_sampled_objective'] if feasible else metrics['ndcg@10'])
                if best_key is None or key > best_key:
                    best_key = key
                    best_row = row
                    save_checkpoint(variant_dir / 'best.pt', model, baseline, variant, step, metrics, cfg)
                report['variants'][variant['name']] = {
                    'settings': variant, 'history': evaluations, 'training': train_logs,
                    'selected': best_row, 'checkpoint': str(variant_dir / 'best.pt'),
                }
                (reports / 'rl.json').write_text(json.dumps(report, indent=2))
                print(f'\n{variant["name"]}: {json.dumps(row)}', flush=True)
                model.eval()
        del model, optimizer
        if device.type == 'mps':
            torch.mps.empty_cache()
    feasible = [(name, value) for name, value in report['variants'].items()
                if not value['settings'].get('sft', False) and value['selected']['meets_ndcg_constraint']]
    report['recommended_variant'] = max(feasible, key=lambda pair: pair[1]['selected']['reward@10'])[0] if feasible else None
    report['elapsed_seconds'] = time.perf_counter() - run_start
    report['limitations'] = 'Reward is predicted by the same frozen model used for optimization. The 5% relevance tolerance is an analysis constraint, not a statistical significance test. One seed; no online or causal retention claim.'
    (reports / 'rl.json').write_text(json.dumps(report, indent=2))
    print('\nValidation comparison (selected checkpoints):', flush=True)
    print(f'{"variant":20} {"NDCG@10":>10} {"reward@10":>10} {"KL":>10} {"valid":>7}', flush=True)
    print(f'{"baseline":20} {baseline_metrics["ndcg@10"]:10.5f} {baseline_metrics["reward@10"]:10.4f} {0.:10.4f} {"yes":>7}')
    for name, value in report['variants'].items():
        row = value['selected']
        print(f'{name:20} {row["ndcg@10"]:10.5f} {row["reward@10"]:10.4f} {row["kl_to_reference"]:10.4f} {str(row["meets_ndcg_constraint"]):>7}')
    print(f'Saved {reports / "rl.json"}. Recommended RL variant:', report['recommended_variant'], flush=True)


if __name__ == '__main__':
    main()
