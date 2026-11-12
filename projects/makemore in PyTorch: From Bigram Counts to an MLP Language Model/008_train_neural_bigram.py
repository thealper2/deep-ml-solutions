import torch
import torch.nn.functional as F

def train_neural_bigram(xs, ys, vocab_size, steps, lr, reg, seed):
    """Full-batch gradient descent on one linear layer; returns (W, losses)."""
    g = torch.Generator()
    g.manual_seed(seed)
    W = torch.randn((vocab_size, vocab_size), generator=g, requires_grad=True)
    losses = []
    n = len(xs)
    for _ in range(steps):
        x_onehot = torch.zeros(n, vocab_size)
        x_onehot[torch.arange(n), xs] = 1.0
        logits = x_onehot @ W
        probs = torch.softmax(logits, dim=1)
        nll = -torch.log(probs[torch.arange(n), ys]).mean()
        loss = nll + reg * (W ** 2).mean()
        losses.append(float(loss.item()))
        if W.grad is not None:
            W.grad.zero_()
        loss.backward()
        with torch.no_grad():
            W -= lr * W.grad
    return W, losses