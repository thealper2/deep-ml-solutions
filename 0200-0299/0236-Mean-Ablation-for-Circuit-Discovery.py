import numpy as np

def mean_ablate(activations: np.ndarray, mask: np.ndarray, means: np.ndarray) -> np.ndarray:
    """
    Apply mean ablation to node activations.
    
    Args:
        activations: Original node activations, shape (n,) or (batch, n)
        mask: Binary mask where 1 = ablate (replace with mean), 0 = keep original
        means: Precomputed mean activations for each node, shape (n,)
    
    Returns:
        Ablated activations with same shape as input
    """
    result = activations.copy()
    result[mask == 1] = means[mask == 1]
    return result
