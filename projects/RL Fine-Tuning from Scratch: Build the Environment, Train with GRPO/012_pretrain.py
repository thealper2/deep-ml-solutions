import random
import torch


def pretrain(policy, demos, steps, batch_size=64, lr=1e-3, seed=0):
    rng = random.Random(seed)
    opt = torch.optim.AdamW(policy.parameters(), lr=lr)
    n = min(batch_size, len(demos))

    loss = 0.0
    for _ in range(steps):
        batch = rng.sample(demos, n)
        loss = sft_loss(policy, batch)
        opt.zero_grad()
        loss.backward()
        opt.step()

    return float(loss)