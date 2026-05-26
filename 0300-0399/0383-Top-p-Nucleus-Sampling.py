import numpy as np

def top_p_sampling(logits: list[float], p: float) -> list[float]:
    """
    Apply top-p (nucleus) sampling to filter a probability distribution.
    
    Args:
        logits: Raw unnormalized scores for each token
        p: Cumulative probability threshold (0 < p <= 1)
    
    Returns:
        Filtered and renormalized probability distribution as a list of floats
    """
    max_logit = max(logits)
    exp_logits = [np.exp(l - max_logit) for l in logits]
    probs = [e / sum(exp_logits) for e in exp_logits]
    indices = list(range(len(probs)))
    sorted_indices = sorted(indices, key=lambda i: probs[i], reverse=True)
    cumsum = 0.0
    nucleus_indices = set()
    for idx in sorted_indices:
        cumsum += probs[idx]
        nucleus_indices.add(idx)
        if cumsum >= p:
            break

    filtered_probs = [0.0] * len(probs)
    for idx in nucleus_indices:
        filtered_probs[idx] = probs[idx]

    total = sum(filtered_probs)
    if total > 0:
        filtered_probs = [prob / total for prob in filtered_probs]

    return np.round(filtered_probs, 4).tolist()