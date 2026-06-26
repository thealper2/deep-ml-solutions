import numpy as np

def mse_loss_on_chosen_action(predicted_q, action_indices, target_q):
    """MSE between Q(s, a_taken) and the bootstrapped target Q."""
    chosen_q = predicted_q[np.arange(len(action_indices)), action_indices]
    return np.mean((chosen_q - target_q) ** 2)
