import torch


def rlvr_experiment(seed=0, sft_steps=600, rl_steps=300, hack_steps=50, cfg=None, k=8, window=50):
    if cfg is None:
        cfg = dict(batch=16, group=8, lr=3e-4, beta=0.02, clip_eps=0.2, normalize_std=True, epochs=1)

    base = build_base(seed, sft_steps)
    base_eval = evaluate(base["policy"], base["held"], k, torch.Generator().manual_seed(seed))

    strict = run_grpo(base, rl_steps, cfg, seed, strict=True)
    strict_eval = evaluate(strict["policy"], base["held"], k, torch.Generator().manual_seed(seed))

    hack = run_grpo(base, hack_steps, cfg, seed, strict=False)
    hack_eval = evaluate(hack["policy"], base["held"], k, torch.Generator().manual_seed(seed))

    history = strict["history"]
    curve = []
    for start in range(0, len(history), window):
        chunk = history[start:start + window]
        end = chunk[-1]["step"] + 1
        reward_mean = sum(h["reward"] for h in chunk) / len(chunk)
        kl_mean = sum(h["kl"] for h in chunk) / len(chunk)
        curve.append({"step": end, "reward": reward_mean, "kl": kl_mean})

    return {
        "base": {"sft_loss": base["sft_loss"], "eval": base_eval},
        "strict": {"history": history, "eval": strict_eval, "curve": curve},
        "hack": {"history": hack["history"], "eval": hack_eval},
        "cfg": cfg,
        "n_train": len(base["train"]),
        "n_held": len(base["held"]),
    }