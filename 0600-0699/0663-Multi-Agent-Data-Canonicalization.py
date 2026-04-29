import numpy as np

def canonicalize_observations(observations: np.ndarray, agent_ids: np.ndarray) -> np.ndarray:
    """
    Canonicalize multi-agent observations for experience sharing.
    
    Args:
        observations: (B, N, F) array of observations
        agent_ids: (B,) array indicating the self-agent index per observation
    
    Returns:
        (B, N, F) array of canonicalized observations, rounded to 4 decimals
    """
    B, N, F = observations.shape
    canonicalized = np.zeros_like(observations)

    for b in range(B):
        obs = observations[b]
        self_idx = agent_ids[b]
        self_features = obs[self_idx]
        other_indices = [i for i in range(N) if i != self_idx]
        other_features = obs[other_indices]
        other_list = [tuple(features) for features in other_features]
        sorted_indices = sorted(range(len(other_list)), key=lambda i: other_list[i])
        sorted_other = other_features[sorted_indices]
        canonicalized[b, 0] = self_features
        canonicalized[b, 1:] = sorted_other

    canonicalized = np.round(canonicalized, 4)
    return canonicalized