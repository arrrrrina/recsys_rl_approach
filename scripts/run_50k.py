"""Prepare a nested 50k train sample, train on MPS, compare with the 10k run."""
from _bootstrap import bootstrap
bootstrap()
from pathlib import Path
import argparse
import json
import subprocess
import sys


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=['mps', 'cpu', 'cuda'], default='mps')
    parser.add_argument('--skip-prepare', action='store_true')
    args = parser.parse_args(argv)
    project = Path(__file__).resolve().parents[1]
    run_dir = Path('experiments/users50k')

    def run(script, *arguments):
        subprocess.run([sys.executable, str(project / 'scripts' / script), *arguments],
                       cwd=project, check=True)

    if not args.skip_prepare:
        run('prepare_data.py', '--config', 'configs/users50k.json',
            '--run-dir', str(run_dir), '--freeze-eval-from', '.')
    run('train_baseline.py', '--config', 'configs/users50k.json',
        '--run-dir', str(run_dir), '--device', args.device)
    run('train_reward.py', '--run-dir', str(run_dir), '--device', args.device)
    old = project / 'reports'
    new = project / run_dir / 'reports'
    comparison = {}
    for label, folder in (('users10k', old), ('users50k', new)):
        baseline = json.loads((folder / 'baseline.json').read_text())
        reward = json.loads((folder / 'reward.json').read_text())
        best = max(baseline['history'], key=lambda row: row['ndcg@10'])
        comparison[label] = {
            'baseline_best_epoch': best['epoch'],
            'ndcg@10': best['ndcg@10'], 'recall@10': best['recall@10'],
            'catalog_coverage@10': best['catalog_coverage@10'],
            'reward_best_epoch': reward['best_epoch'],
            'reward_rmse': reward['best_validation']['rmse'],
            'reward_item_prior_rmse': reward['controls_validation']['smoothed_item_mean']['rmse'],
            'baseline_optimizer_steps': len(baseline['history']) * (
                (json.loads((folder / 'data_audit.json').read_text())['prepared']['train']['examples']
                 + baseline['config']['batch_size'] - 1) // baseline['config']['batch_size']),
        }
    comparison['note'] = 'Same validation/test arrays and vocabulary. Same architecture and epoch limits, but larger train means more optimizer updates per epoch; not a compute-matched comparison.'
    (new / 'comparison_10k_50k.json').write_text(json.dumps(comparison, indent=2))
    print('\n10k vs 50k:', json.dumps(comparison, indent=2), flush=True)


if __name__ == '__main__':
    main()
