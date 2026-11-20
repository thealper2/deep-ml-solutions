import torch
import torch.nn as nn


class RewardModel(nn.Module):
    def __init__(self, vocab_size, d_model=32, n_head=2, n_layer=2, max_len=16):
        """self.body: a TinyLM. self.head: a linear layer from d_model to one number."""
        super().__init__()
        self.body = TinyLM(vocab_size, d_model, n_head, n_layer, max_len)
        self.head = nn.Linear(d_model, 1)

    def forward(self, ids):
        """(B, T) token ids -> (B,) rewards, read from the hidden state at the last position."""
        h = self.body.hidden(ids)
        return self.head(h[:, -1, :]).squeeze(-1)

    def score(self, prompts, completions):
        """Lists of strings -> (len(prompts),) rewards, with gradients."""
        encoded = [encode_example(p, c)[0] for p, c in zip(prompts, completions)]
        ids = torch.tensor(encoded, dtype=torch.long)
        return self.forward(ids)
