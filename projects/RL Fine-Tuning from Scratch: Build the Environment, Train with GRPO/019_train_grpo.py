import random
import torch
import torch.nn as nn


def train_grpo(policy, ref, tasks, steps, cfg, seed=0, strict=True):
    rng = random.Random(seed)
    generator = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(policy.parameters(), lr=cfg["lr"])

    history = []
    for step in range(steps):
        batch = rng.sample(tasks, cfg["batch"])
        out = rollout(policy, ref, batch, cfg["group"], generator, strict, cfg["normalize_std"])

        reward_mean = float(out["rewards"].mean())
        stats = None
        loss = None
        for _ in range(cfg["epochs"]):
            logp = completion_logprobs(policy, out["seqs"])
            loss, stats = grpo_loss(
                logp,
                out["old_logp"],
                out["ref_logp"],
                out["advantages"],
                out["mask"],
                cfg["clip_eps"],
                cfg["beta"],
            )
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()

        history.append({
            "step": step,
            "reward": reward_mean,
            "loss": float(loss),
            "pg": float(stats["pg"]),
            "kl": float(stats["kl"]),
            "clip_frac": float(stats["clip_frac"]),
        })

    return history