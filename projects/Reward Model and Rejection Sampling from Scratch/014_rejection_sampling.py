import copy
import torch


def rejection_sampling(model, rm, prompts, n=8, temperature=1.0, steps=200, batch_size=32, lr=1e-3, seed=0):
    """Keep the best of n completions per prompt, fine-tune a copy of the model on them; returns (tuned, kept)."""
    g = torch.Generator().manual_seed(seed)
    kept = []
    for prompt in prompts:
        best, _, _ = best_of_n(model, rm, prompt, n, temperature, g)
        kept.append((prompt, best))

    tuned = copy.deepcopy(model)
    instruction_tune(tuned, kept, steps, batch_size, lr, seed)
    return tuned, kept