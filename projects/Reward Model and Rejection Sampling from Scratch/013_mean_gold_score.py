import torch


def mean_gold_score(model, prompts, n=4, temperature=1.0, seed=0):
    """Average gold score over n sampled completions for each prompt, as a Python float."""
    g = torch.Generator().manual_seed(seed)
    total = 0.0
    count = 0
    for prompt in prompts:
        completions = sample_completions(model, prompt, n, temperature, g)
        for c in completions:
            total += gold_score(prompt, c)
            count += 1
    return total / count