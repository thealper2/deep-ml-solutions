import numpy as np

def argmax_action_from_q_values(masked_q_values):
    """Return the index of the largest entry in masked_q_values as an int."""
    return np.argmax(masked_q_values)
