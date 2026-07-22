import numpy as np

def ppo_clip_loss(old_log_probs: np.ndarray, new_log_probs: np.ndarray, advantages: np.ndarray, epsilon: float = 0.2) -> tuple:
    """
    Compute the PPO clipped surrogate loss and clip fraction diagnostic.
    
    Args:
        old_log_probs: Log-probabilities of actions under the old policy, shape (N,)
        new_log_probs: Log-probabilities of actions under the new policy, shape (N,)
        advantages: Advantage estimates for each sample, shape (N,)
        epsilon: Clipping parameter for the probability ratio
    
    Returns:
        Tuple of (loss, clip_fraction), both rounded to 4 decimal places
        - loss: Mean negated clipped surrogate objective (for minimization)
        - clip_fraction: Fraction of samples where the ratio was clipped
    """
    ratio = np.exp(new_log_probs - old_log_probs)
    surr1 = ratio * advantages
    clipped_ratio = np.clip(ratio, 1.0 - epsilon, 1.0 + epsilon)
    surr2 = clipped_ratio * advantages
    loss = -np.mean(np.minimum(surr1, surr2))
    clip_fraction = np.mean((ratio < 1.0 - epsilon) | (ratio > 1.0 + epsilon))
    return loss, clip_fraction
