import numpy as np

def gibbs_softmax_action_selection(q_values: list, temperature: float, seed: int) -> tuple:
    """
    Perform Gibbs softmax (Boltzmann) action selection.

    Args:
        q_values: list of floats, estimated action values
        temperature: float, temperature parameter (tau > 0)
        seed: int, random seed for reproducibility

    Returns:
        tuple: (probabilities as list of floats, selected action as int)
    """
    np.random.seed(seed)
    q = np.array(q_values, dtype=np.float64)
    q_shifted = q - np.max(q)
    q_scaled = q_shifted / temperature
    q_scaled = np.clip(q_scaled, -700, 700)
    exp_q = np.exp(q_scaled)
    probabilities = exp_q / np.sum(exp_q)
    probabilities = probabilities / np.sum(probabilities)
    action = int(np.random.choice(len(q_values), p=probabilities))
    return probabilities.tolist(), action
    