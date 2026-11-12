import torch
import torch.nn.functional as F
import math

def train(model, X, Y, steps, batch_size=32, lr=0.1, seed=0):
    """Minibatch SGD with lr, then lr / 10 for the last quarter. Returns (losses, update-to-data log ratios)."""
    g = torch.Generator()
    g.manual_seed(seed)
    params = model.parameters()
    n = X.shape[0]
    losses = []
    ud = []
    for step in range(steps):
        idx = torch.randint(0, n, (batch_size,), generator=g)
        logits = model(X[idx])
        loss = F.cross_entropy(logits, Y[idx])
        for p in params:
            p.grad = None
        loss.backward()
        cur_lr = lr if step < int(0.75 * steps) else lr / 10
        with torch.no_grad():
            for p in params:
                p -= cur_lr * p.grad
        row = []
        with torch.no_grad():
            for p in params:
                if p.ndim == 2:
                    upd = (cur_lr * p.grad).std().item()
                    wstd = p.data.std().item()
                    row.append(math.log10(upd / wstd))
        losses.append(float(loss.item()))
        ud.append(row)
    return losses, ud