import numpy as np
from typing import List, Tuple

def fit_bradley_terry(comparisons: List[Tuple[int, int]], n_items: int, 
                      learning_rate: float = 0.5, n_iterations: int = 100) -> np.ndarray:
    """
    Fit Bradley-Terry model parameters using maximum likelihood estimation.
    
    Args:
        comparisons: List of (winner_idx, loser_idx) tuples
        n_items: Total number of items to rank
        learning_rate: Step size for gradient ascent
        n_iterations: Number of optimization iterations
    
    Returns:
        np.ndarray: Estimated strength parameters of shape (n_items,)
    """
    beta = np.zeros(n_items)

    if len(comparisons) == 0:
        return beta

    for _ in range(n_iterations):
        grad = np.zeros(n_items)

        for winner, loser in comparisons:
            diff = beta[winner] - beta[loser]
            p = 1.0 / (1.0 + np.exp(-np.clip(diff, -500, 500)))

            grad[winner] += 1.0 - p
            grad[loser] += p - 1.0

        beta += learning_rate * grad

        beta -= beta.mean()

    return beta
