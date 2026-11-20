import torch


def train_reward_model(rm, pairs, steps, batch_size=64, lr=5e-3, seed=0):
    """AdamW on random batches of pairs with pairwise_loss; returns each step's loss as a float."""
    g = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.AdamW(rm.parameters(), lr=lr)
    rm.train()

    losses = []
    n = len(pairs)
    for _ in range(steps):
        ix = torch.randint(0, n, (batch_size,), generator=g).tolist()
        batch = [pairs[i] for i in ix]
        prompts = [b[0] for b in batch]
        chosen = [b[1] for b in batch]
        rejected = [b[2] for b in batch]

        r_chosen = rm.score(prompts, chosen)
        r_rejected = rm.score(prompts, rejected)
        loss = pairwise_loss(r_chosen, r_rejected)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses