import torch


def train_sft(model, ids, mask, steps, batch_size=32, lr=3e-3, generator=None):
    """AdamW on random batches of trajectories; returns each step's loss as a float."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    N = ids.shape[0]
    losses = []
    for _ in range(steps):
        rows = torch.randint(0, N, (batch_size,), generator=generator)
        logits = model(ids[rows])
        loss = sft_loss(logits, ids[rows], mask[rows])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
    return losses