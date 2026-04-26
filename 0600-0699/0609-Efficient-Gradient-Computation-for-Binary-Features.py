import numpy as np

def efficient_binary_td_update(initial_weights: list, transitions: list, alpha: float, gamma: float) -> np.ndarray:
    """
    Perform TD(0) updates with binary feature representations.
    
    Args:
        initial_weights: Initial weight vector
        transitions: List of (active_s, reward, active_s_next, done) tuples
        alpha: Step size for updates
        gamma: Discount factor
    
    Returns:
        Final weight vector as numpy array
    """
    weights = np.array(initial_weights, dtype=np.float64)
    for active_s, reward, active_s_next, done in transitions:
        v_s = np.sum(weights[active_s]) if active_s else 0.0
        if done or not active_s_next:
            v_s_next = 0.0
        else:
            v_s_next = np.sum(weights[active_s_next])

        td_error = reward + gamma * v_s_next - v_s
        for idx in active_s:
            weights[idx] += alpha * td_error

    return weights.tolist()