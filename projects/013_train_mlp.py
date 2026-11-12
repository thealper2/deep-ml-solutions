import torch
import torch.nn.functional as F

def train_mlp(model, X, Y, steps, batch_size, lr, seed):
    """Minibatch gradient descent with lr, then lr / 10 for the second half; returns the losses."""
    g = torch.Generator()
    g.manual_seed(seed)
    n = X.shape[0]
    losses = []
    params = model.parameters()
    for step in range(steps):
        idx = torch.randint(0, n, (batch_size,), generator=g)
        logits = model(X[idx])
        loss = F.cross_entropy(logits, Y[idx])
        for p in params:
            p.grad = None
        loss.backward()
        cur_lr = lr if step < steps // 2 else lr / 10
        with torch.no_grad():
            for p in params:
                p -= cur_lr * p.grad
        losses.append(float(loss.item()))
    return losses