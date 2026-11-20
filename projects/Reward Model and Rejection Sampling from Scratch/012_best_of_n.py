import torch


def best_of_n(model, rm, prompt, n, temperature=1.0, generator=None):
    """Sample n completions, score them with rm, return (best, completions, scores)."""
    completions = sample_completions(model, prompt, n, temperature, generator)
    rm.eval()
    with torch.no_grad():
        scores = rm.score([prompt] * n, completions)
    best_idx = int(torch.argmax(scores).item())
    return completions[best_idx], completions, scores