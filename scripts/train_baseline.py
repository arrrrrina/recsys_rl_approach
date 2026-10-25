"""Train GPT-2 baseline on MPS; choose checkpoint using validation only."""
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

from recsys_rl.model import GPTRec
from recsys_rl.runtime import resolve_device, seed_everything, synchronize
from recsys_rl.evaluation import PreparedDataset, evaluate, evaluate_popularity, move_context


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=Path('configs/small.json'))
    parser.add_argument('--run-dir', type=Path, default=Path('.'))
    parser.add_argument('--device', choices=['mps', 'cpu', 'cuda'], default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--evaluate-test', action='store_true', help='Use only after final model selection')
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text())
    for argument, key in ((args.device, 'device'), (args.epochs, 'epochs'), (args.batch_size, 'batch_size')):
        if argument is not None:
            config[key] = argument
    device = resolve_device(config['device'])
    seed_everything(config['seed'])
    prepared = args.run_dir / 'artifacts/prepared'
    output = args.run_dir / 'artifacts/baseline'
    reports = args.run_dir / 'reports'
    reports.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    audit = json.loads((reports / 'data_audit.json').read_text())
    for key in ('seed', 'max_users', 'min_train_days', 'max_length', 'train_cutoff', 'validation_cutoff'):
        if config[key] != audit['config'][key]:
            raise ValueError(f'Prepared data config mismatch: {key}; rerun prepare_data.py')
    train = PreparedDataset(prepared / 'train.npz')
    validation = PreparedDataset(prepared / 'validation.npz')
    popularity = np.load(prepared / 'train_popularity.npy')
    model = GPTRec(len(popularity), config).to(device=device, dtype=torch.float32)
    print(f'Device={device}; parameters={sum(p.numel() for p in model.parameters()):,}; '
          f'train={len(train):,}; validation={len(validation):,}', flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'],
                                 weight_decay=config['weight_decay'], foreach=False)
    loader = DataLoader(train, batch_size=config['batch_size'], shuffle=True, num_workers=0)
    popular_metrics = evaluate_popularity(validation, popularity)
    print('Popularity validation:', json.dumps(popular_metrics), flush=True)
    history = []
    best_score = -float('inf')
    stale_epochs = 0
    run_start = time.perf_counter()
    for epoch in range(1, config['epochs'] + 1):
        model.train()
        loss_sum = 0.0
        start = time.perf_counter()
        progress = tqdm(loader, desc=f'Train {epoch}/{config["epochs"]}')
        for batch in progress:
            inputs, lengths = move_context(batch, device)
            targets = batch['targets'].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs, lengths)
            loss = torch.nn.functional.cross_entropy(logits, targets)
            if not torch.isfinite(loss).item():
                raise RuntimeError('Non-finite baseline loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'], foreach=False)
            optimizer.step()
            loss_sum += loss.item() * len(targets)
            progress.set_postfix(loss=f'{loss.item():.4f}')
        metrics = evaluate(model, validation, device, config['eval_batch_size'], popularity)
        synchronize(device)
        row = {'epoch': epoch, 'train_loss': loss_sum / len(train),
               'seconds': time.perf_counter() - start, **metrics}
        history.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if metrics['ndcg@10'] > best_score:
            best_score = metrics['ndcg@10']
            stale_epochs = 0
            # CPU checkpoint is portable to MPS/CPU and later RL training.
            state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            torch.save({'state_dict': state, 'config': config, 'vocab_size': len(popularity),
                        'epoch': epoch, 'validation': metrics,
                        'data_provenance': audit['provenance']}, output / 'best.pt')
        else:
            stale_epochs += 1
        report = {'config': config, 'device': str(device), 'popularity_validation': popular_metrics,
                  'history': history, 'best_validation_ndcg@10': best_score,
                  'elapsed_seconds': time.perf_counter() - run_start}
        (reports / 'baseline.json').write_text(json.dumps(report, indent=2))
        if stale_epochs >= config['patience']:
            print('Early stopping', flush=True)
            break
    if args.evaluate_test:
        checkpoint = torch.load(output / 'best.pt', map_location='cpu', weights_only=True)
        model.load_state_dict(checkpoint['state_dict'])
        test = PreparedDataset(prepared / 'test.npz')
        report['test'] = evaluate(model, test, device, config['eval_batch_size'], popularity, 'Test')
        report['popularity_test'] = evaluate_popularity(test, popularity)
        (reports / 'baseline.json').write_text(json.dumps(report, indent=2))
    print(f'Saved {output / "best.pt"} and {reports / "baseline.json"}', flush=True)


if __name__ == '__main__':
    main()
