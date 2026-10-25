"""Full-catalog next-event metrics, macro averaged over users."""
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class PreparedDataset(Dataset):
    def __init__(self, path):
        with np.load(path) as archive:
            self.arrays = {key: archive[key] for key in archive.files}

    def __len__(self):
        return len(self.arrays['targets'])

    def __getitem__(self, index):
        return {key: torch.as_tensor(value[index]) for key, value in self.arrays.items()}


def move_context(batch, device):
    # Trim right padding per batch to reduce attention work. No float64 on MPS.
    width = int(batch['lengths'].max())
    return batch['input_ids'][:, :width].to(device), batch['lengths'].to(device)


def next_item_metrics(predictions, targets, user_ids, ks=(5, 10, 20)):
    predictions, targets, user_ids = map(np.asarray, (predictions, targets, user_ids))
    if len(targets) == 0:
        raise ValueError('Cannot evaluate an empty split')
    _, inverse = np.unique(user_ids, return_inverse=True)
    counts = np.bincount(inverse)
    result = {}
    for k in ks:
        hits = predictions[:, :k] == targets[:, None]
        found = hits.any(axis=1)
        ranks = hits.argmax(axis=1) + 1
        for name, values in (
            ('recall', found.astype(float)),
            ('ndcg', found / np.log2(ranks + 1)),
        ):
            by_user = np.bincount(inverse, weights=values) / counts
            result[f'{name}@{k}'] = float(by_user.mean())
    result['examples'] = len(targets)
    result['users'] = len(counts)
    return result


@torch.no_grad()
def evaluate(model, dataset, device, batch_size=256, popularity=None, description='Validation',
             return_details=False):
    from tqdm.auto import tqdm
    model.eval()
    predictions, targets, users = [], [], []
    total_loss = 0.0
    repeats = []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    for batch in tqdm(loader, desc=description):
        inputs, lengths = move_context(batch, device)
        logits = model(inputs, lengths)
        target = batch['targets'].to(device)
        total_loss += torch.nn.functional.cross_entropy(logits, target, reduction='sum').item()
        recs = logits.topk(min(20, logits.shape[1] - 1), dim=1).indices.cpu().numpy()
        predictions.append(recs)
        targets.append(batch['targets'].numpy())
        users.append(batch['user_ids'].numpy())
        histories = batch['input_ids'].numpy()
        repeats.extend((recs[:, :10, None] == histories[:, None, :]).any(axis=2).mean(axis=1))
    predictions = np.concatenate(predictions)
    metrics = next_item_metrics(predictions, np.concatenate(targets), np.concatenate(users))
    metrics['cross_entropy'] = total_loss / len(dataset)
    top10 = predictions[:, :10].reshape(-1)
    frequencies = np.bincount(top10, minlength=logits.shape[1])[1:]
    probabilities = frequencies[frequencies > 0] / frequencies.sum()
    metrics['catalog_coverage@10'] = float((frequencies > 0).mean())
    metrics['catalog_entropy@10'] = float(-(probabilities * np.log(probabilities)).sum())
    metrics['largest_item_share@10'] = float(frequencies.max() / frequencies.sum())
    metrics['seen_in_last_context_share@10'] = float(np.mean(repeats))
    if popularity is not None:
        metrics['mean_train_popularity@10'] = float(popularity[top10].mean())
    if return_details:
        return metrics, {'predictions': predictions, 'targets': np.concatenate(targets),
                         'user_ids': np.concatenate(users)}
    return metrics


def evaluate_popularity(dataset, popularity):
    ranking = np.argsort(-popularity[1:], kind='stable')[:20] + 1
    predictions = np.broadcast_to(ranking, (len(dataset), len(ranking)))
    return next_item_metrics(predictions, dataset.arrays['targets'], dataset.arrays['user_ids'])
