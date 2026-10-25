import random
import numpy as np
import torch


def resolve_device(name):
    if name == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError(
            'MPS недоступен этому процессу. Запустите команду в своём Terminal '
            'с Python/PyTorch для Apple Silicon. Для отладки явно передайте --device cpu.'
        )
    if name == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    return torch.device(name)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def synchronize(device):
    if device.type == 'mps':
        torch.mps.synchronize()
    elif device.type == 'cuda':
        torch.cuda.synchronize()
