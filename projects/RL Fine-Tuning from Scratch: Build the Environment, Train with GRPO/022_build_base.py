import random
import torch


def build_base(seed=0, sft_steps=600, n_draws=4000, n_held=200):
    rng = random.Random(seed)
    train, held = task_split(rng, n_draws, n_held)

    demos = make_demos(rng, train)

    torch.manual_seed(seed)
    policy = Policy()

    sft_loss = pretrain(policy, demos, sft_steps, seed=seed)

    return {
        "policy": policy,
        "train": train,
        "held": held,
        "sft_loss": sft_loss,
    }