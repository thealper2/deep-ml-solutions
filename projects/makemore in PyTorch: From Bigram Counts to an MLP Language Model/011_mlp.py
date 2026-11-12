import torch

class MLP:
    def __init__(self, vocab_size, block_size, emb_dim, hidden, seed):
        """Draw C, W1, b1, W2, b2 (in that order) from one seeded generator; all require grad."""
        self.block_size = block_size
        g = torch.Generator()
        g.manual_seed(seed)
        self.C = torch.randn((vocab_size, emb_dim), generator=g, requires_grad=True)
        self.W1 = torch.randn((block_size * emb_dim, hidden), generator=g, requires_grad=True)
        self.b1 = torch.randn((hidden,), generator=g, requires_grad=True)
        self.W2 = torch.randn((hidden, vocab_size), generator=g, requires_grad=True)
        self.b2 = torch.randn((vocab_size,), generator=g, requires_grad=True)

    def parameters(self):
        """[C, W1, b1, W2, b2]"""
        return [self.C, self.W1, self.b1, self.W2, self.b2]

    def __call__(self, X):
        """(n, block_size) ids -> (n, vocab_size) logits."""
        emb = self.C[X]
        flat = emb.reshape(X.shape[0], -1)
        h = torch.tanh(flat @ self.W1 + self.b1)
        return h @ self.W2 + self.b2