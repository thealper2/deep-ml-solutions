import copy


def run_grpo(base, steps, cfg, seed=0, strict=True):
    policy = copy.deepcopy(base["policy"])
    ref = copy.deepcopy(base["policy"])
    history = train_grpo(policy, ref, base["train"], steps, cfg, seed, strict)
    return {
        "policy": policy,
        "ref": ref,
        "history": history,
    }