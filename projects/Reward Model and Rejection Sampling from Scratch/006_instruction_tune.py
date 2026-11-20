import torch


def instruction_tune(model, examples, steps, batch_size=32, lr=3e-3, seed=0):
    """AdamW on random batches of encoded examples with sft_loss; returns each step's loss as a float."""
    encoded = [encode_example(p, c) for p, c in examples]
    ids = torch.tensor([e[0] for e in encoded], dtype=torch.long)
    mask = torch.tensor([e[1] for e in encoded], dtype=torch.float32)

    g = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()

    losses = []
    n = len(examples)
    for _ in range(steps):
        ix = torch.randint(0, n, (batch_size,), generator=g)
        logits = model(ids[ix])
        loss = sft_loss(logits, ids[ix], mask[ix])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return losses
