import numpy as np

def quality_filter_rejection_sampling(scores: list, threshold: float, n_select: int = None) -> dict:
    """
    Filter generated samples using quality-based rejection sampling.
    
    Args:
        scores: list of float quality scores for generated candidate samples
        threshold: minimum quality score required for acceptance
        n_select: optional maximum number of samples to return (top by score)
    
    Returns:
        dict with 'accepted_indices', 'acceptance_rate', 'mean_quality'
    """
    filtered = [(i, score) for i, score in enumerate(scores) if score >= threshold]

    acceptance_rate = len(filtered) / len(scores)

    if n_select is None:
        n_select = len(filtered)
    else:
        n_select = min(n_select, len(filtered))

    if len(filtered) == 0:
        return {
            'accepted_indices': [],
            'acceptance_rate': round(acceptance_rate, 4),
            'mean_quality': 0.0,
        }

    filtered_sorted = sorted(filtered, key=lambda x: x[1], reverse=True)
    top_selected = filtered_sorted[:n_select]

    accepted_indices = [idx for idx, _ in top_selected]

    mean_quality = sum([score for _, score in top_selected]) / len(top_selected)
    
    return {
        'accepted_indices': accepted_indices,
        'acceptance_rate': round(acceptance_rate, 4),
        'mean_quality': round(mean_quality, 4),
    }