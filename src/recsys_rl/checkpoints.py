from pathlib import Path
import torch
from recsys_rl.model import GPTRec
from recsys_rl.reward import RewardModel


def load_policy(path, device):
    checkpoint = torch.load(Path(path), map_location='cpu', weights_only=True)
    model = GPTRec(checkpoint['vocab_size'], checkpoint['config'])
    model.load_state_dict(checkpoint['state_dict'])
    return model.to(device), checkpoint


def load_reward(path, device):
    checkpoint = torch.load(Path(path), map_location='cpu', weights_only=True)
    state = checkpoint['state_dict']
    model = RewardModel(state['embedding.weight'], checkpoint['item_prior'],
                        checkpoint['config']['dropout'])
    model.load_state_dict(state)
    model.to(device).eval().requires_grad_(False)
    return model, checkpoint
