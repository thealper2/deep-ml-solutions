import numpy as np

def apply_steering_vector(hidden_states, steering_vector, coefficient, last_n_tokens=None, normalize=False):
    """
    Apply a steering vector to hidden states at inference time.
    """
    hidden_states = np.array(hidden_states)
    steering_vector = np.array(steering_vector)

    if normalize:
        norm = np.linalg.norm(steering_vector)
        if norm > 0:
            steering_vector = steering_vector / norm

    seq_len = hidden_states.shape[0]
    
    if last_n_tokens is None:
        start_idx = 0
    else:
        start_idx = max(0, seq_len - last_n_tokens)

    modification = coefficient * steering_vector
    for i in range(start_idx, seq_len):
        hidden_states[i] = hidden_states[i] + modification

    return hidden_states.tolist()