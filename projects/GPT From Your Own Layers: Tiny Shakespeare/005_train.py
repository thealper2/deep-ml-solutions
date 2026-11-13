import torch

def train(model, data, block_size, batch_size, steps, lr=1e-3, generator=None):
    """AdamW training on random batches; returns each step's loss as a float."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    losses = []
    for _ in range(steps):
        xb, yb = get_batch(data, block_size, batch_size, generator=generator)
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
    return losses