"""Fit reward regression on observed train events, select on validation RMSE."""
from _bootstrap import bootstrap
bootstrap()
from pathlib import Path
import argparse
import json
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from recsys_rl.evaluation import PreparedDataset, move_context
from recsys_rl.runtime import resolve_device, seed_everything
from recsys_rl.reward import RewardModel, fit_item_prior, regression_metrics, calibration_bins


@torch.no_grad()
def predict(model, dataset, device, batch_size):
    model.eval()
    predictions, targets, items = [], [], []
    for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0):
        inputs, lengths = move_context(batch, device)
        predictions.append(model(inputs, lengths, batch['targets'].to(device)).cpu().numpy())
        targets.append(batch['rewards'].numpy())
        items.append(batch['targets'].numpy())
    return np.concatenate(targets), np.concatenate(predictions), np.concatenate(items)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=Path('configs/reward.json'))
    parser.add_argument('--run-dir', type=Path, default=Path('.'))
    parser.add_argument('--device', choices=['mps', 'cpu', 'cuda'], default=None)
    args = parser.parse_args(argv)
    cfg = json.loads(args.config.read_text())
    if args.device is not None:
        cfg['device'] = args.device
    device = resolve_device(cfg['device'])
    seed_everything(cfg['seed'])
    prepared = args.run_dir / 'artifacts/prepared'
    output = args.run_dir / 'artifacts/reward'
    reports = args.run_dir / 'reports'
    reports.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    baseline = torch.load(args.run_dir / 'artifacts/baseline/best.pt', map_location='cpu', weights_only=True)
    audit = json.loads((reports / 'data_audit.json').read_text())
    if baseline['data_provenance'] != audit['provenance']:
        raise ValueError('Baseline and prepared data provenance differ')
    for key in ('seed', 'max_users', 'min_train_days', 'max_length', 'train_cutoff', 'validation_cutoff'):
        if baseline['config'][key] != audit['config'][key]:
            raise ValueError(f'Baseline and prepared data config differ: {key}')
    train_all = PreparedDataset(prepared / 'train.npz')
    validation_all = PreparedDataset(prepared / 'validation.npz')
    train = Subset(train_all, np.flatnonzero(np.isfinite(train_all.arrays['rewards'])).tolist())
    validation = Subset(validation_all, np.flatnonzero(np.isfinite(validation_all.arrays['rewards'])).tolist())
    if not len(train) or not len(validation):
        raise ValueError('Reward train/validation is empty')
    prior, counts, mean = fit_item_prior(train_all.arrays['targets'], train_all.arrays['rewards'],
                                        baseline['vocab_size'], cfg['prior_smoothing'])
    embeddings = baseline['state_dict']['transformer.wte.weight']
    model = RewardModel(embeddings, prior, cfg['dropout']).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'],
                                 weight_decay=cfg['weight_decay'], foreach=False)
    loader = DataLoader(train, batch_size=cfg['batch_size'], shuffle=True, num_workers=0)
    labels = validation_all.arrays['rewards']
    valid = np.isfinite(labels)
    labels = labels[valid]
    target_items = validation_all.arrays['targets'][valid]
    controls = {
        'constant_train_mean': regression_metrics(labels, np.full_like(labels, mean)),
        'smoothed_item_mean': regression_metrics(labels, prior[target_items]),
    }
    print(f'Device={device}; train={len(train):,}; validation={len(validation):,}', flush=True)
    print('Control validation:', json.dumps(controls), flush=True)
    history = []
    best_rmse = float('inf')
    stale = 0
    start = time.perf_counter()
    for epoch in range(1, cfg['epochs'] + 1):
        model.train()
        loss_sum = 0.0
        progress = tqdm(loader, desc=f'Reward {epoch}/{cfg["epochs"]}')
        for batch in progress:
            inputs, lengths = move_context(batch, device)
            target = batch['rewards'].to(device=device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(inputs, lengths, batch['targets'].to(device))
            loss = torch.nn.functional.mse_loss(prediction, target)
            if not torch.isfinite(loss).item():
                raise RuntimeError('Non-finite reward loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['gradient_clip'], foreach=False)
            optimizer.step()
            loss_sum += loss.item() * len(target)
            progress.set_postfix(mse=f'{loss.item():.4f}')
        y, prediction, items = predict(model, validation, device, cfg['eval_batch_size'])
        metrics = regression_metrics(y, prediction)
        row = {'epoch': epoch, 'train_mse': loss_sum / len(train), **metrics}
        history.append(row)
        print(json.dumps(row), flush=True)
        if metrics['rmse'] < best_rmse:
            best_rmse = metrics['rmse']
            stale = 0
            torch.save({
                'state_dict': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                'config': cfg, 'vocab_size': baseline['vocab_size'], 'embedding_dim': embeddings.shape[1],
                'epoch': epoch, 'validation': metrics, 'global_train_mean': mean,
                'item_prior': torch.from_numpy(prior),
                'data_provenance': audit['provenance'],
            }, output / 'best.pt')
        else:
            stale += 1
        if stale >= cfg['patience']:
            print('Early stopping', flush=True)
            break
    checkpoint = torch.load(output / 'best.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['state_dict'])
    y, prediction, items = predict(model, validation, device, cfg['eval_batch_size'])
    slices = {}
    for name, mask in {
        'train_target_count_0_4': counts[items] < 5,
        'train_target_count_5_49': (counts[items] >= 5) & (counts[items] < 50),
        'train_target_count_50_plus': counts[items] >= 50,
    }.items():
        if mask.any():
            slices[name] = regression_metrics(y[mask], prediction[mask])
    best_metrics = regression_metrics(y, prediction)
    report = {
        'config': cfg, 'history': history, 'best_epoch': checkpoint['epoch'],
        'best_validation': best_metrics, 'controls_validation': controls,
        'improves_constant_rmse': best_metrics['rmse'] < controls['constant_train_mean']['rmse'],
        'improves_item_prior_rmse': best_metrics['rmse'] < controls['smoothed_item_mean']['rmse'],
        'calibration': calibration_bins(y, prediction), 'frequency_slices': slices,
        'elapsed_seconds': time.perf_counter() - start,
        'limitations': 'Observed interactions only; no exposure negatives. Scores on unobserved recommendations are extrapolations. Low-count candidate predictions are less reliable. No test-based selection.',
    }
    (reports / 'reward.json').write_text(json.dumps(report, indent=2))
    np.savez_compressed(output / 'validation_predictions.npz', targets=y, predictions=prediction, items=items)
    print('Best validation:', json.dumps(best_metrics), flush=True)
    print(f'Saved {output / "best.pt"} and {reports / "reward.json"}', flush=True)


if __name__ == '__main__':
    main()
