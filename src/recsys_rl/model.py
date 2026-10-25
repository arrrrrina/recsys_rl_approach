"""GPT-2 next-item policy adapted from references/GPTRec_standalone.ipynb."""
import torch
from torch import nn


class GPTRec(nn.Module):
    def __init__(self, vocab_size, config):
        super().__init__()
        from transformers import GPT2Config, GPT2Model
        gpt_config = GPT2Config(
            vocab_size=vocab_size,
            n_positions=config['max_length'],
            n_ctx=config['max_length'],
            n_embd=config['n_embd'],
            n_layer=config['n_layer'],
            n_head=config['n_head'],
            resid_pdrop=config.get('dropout', 0.1),
            embd_pdrop=config.get('dropout', 0.1),
            attn_pdrop=config.get('dropout', 0.1),
            pad_token_id=0,
            use_cache=False,
        )
        # Eager attention keeps the small model independent of fused GPU kernels.
        gpt_config._attn_implementation = 'eager'
        self.transformer = GPT2Model(gpt_config)
        self.head = nn.Linear(config['n_embd'], vocab_size, bias=False)
        self.head.weight = self.transformer.wte.weight

    def forward(self, input_ids, lengths):
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        mask = positions.unsqueeze(0) < lengths.unsqueeze(1)
        hidden = self.transformer(input_ids=input_ids, attention_mask=mask).last_hidden_state
        last = hidden[torch.arange(len(lengths), device=hidden.device), lengths - 1]
        logits = self.head(last)
        # PAD must never be an action or target. A finite value is safe for later KL.
        return torch.cat([torch.full_like(logits[:, :1], -1e9), logits[:, 1:]], dim=-1)
