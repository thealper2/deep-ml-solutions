import numpy as np

def weighted_importance_sampling(returns: list, target_probs: list, behavior_probs: list) -> tuple:
    """
    Compute ordinary and weighted importance sampling estimates.

    Args:
        returns: List of episode returns (floats), one per episode
        target_probs: List of lists; target_probs[i][t] is the target policy
                      probability for the action taken at step t of episode i
        behavior_probs: List of lists; behavior_probs[i][t] is the behavior policy
                        probability for the action taken at step t of episode i

    Returns:
        Tuple of (ordinary_is_estimate, weighted_is_estimate), each rounded to 4 decimals
    """
    ratios = []
    for ep_returns, t_probs, b_probs in zip(returns, target_probs, behavior_probs):
        ratio = 1.0
        for tp, bp in zip(t_probs, b_probs):
            if bp == 0:
                return (0.0, 0.0)
            
            ratio *= (tp / bp)

        ratios.append(ratio)

    ordinary_is = sum(r * ret for r, ret in zip(ratios, returns)) / len(returns)

    total_ratio = sum(ratios)
    if total_ratio == 0:
        weighted_is = 0.0
    else:
        weighted_is = sum(r * ret for r, ret in zip(ratios, returns)) / total_ratio

    return round(ordinary_is, 4), round(weighted_is, 4)
