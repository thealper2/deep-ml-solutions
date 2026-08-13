import numpy as np

def td_backprop_update(state, next_state, reward, gamma, alpha, W1, b1, W2, b2, terminal=False):
    """
    Perform one semi-gradient TD(0) update step with a neural network value function.
    
    Args:
        state: Current state vector, shape (d,)
        next_state: Next state vector, shape (d,)
        reward: Scalar reward
        gamma: Discount factor
        alpha: Learning rate
        W1: Hidden layer weights, shape (d, h)
        b1: Hidden layer bias, shape (h,)
        W2: Output layer weights, shape (h, 1)
        b2: Output layer bias, shape (1,)
        terminal: Whether next_state is terminal
    
    Returns:
        Tuple of (td_error, W1_new, b1_new, W2_new, b2_new)
    """
    state = np.array(state)
    d = state.shape[0]
    h = W1.shape[1]
    
    z1 = state @ W1 + b1
    a1 = np.maximum(0, z1)
    V_s = V_s = (a1 @ W2 + b2).item()
    
    if next_state is None:
        V_next = 0.0
    else:
        next_state = np.array(next_state)
        z1_next = next_state @ W1 + b1
        a1_next = np.maximum(0, z1_next)
        V_next = (a1_next @ W2 + b2).item()
    
    td_error = reward + gamma * V_next - V_s
    
    relu_mask = (z1 > 0).astype(float)
    
    grad_W2 = np.outer(a1, np.array([1.0]))
    grad_b2 = np.array([1.0])
    
    grad_z1 = W2.flatten() * relu_mask
    
    grad_W1 = np.outer(state, grad_z1)
    grad_b1 = grad_z1
    
    updated_W1 = np.round(W1 + alpha * td_error * grad_W1, 4)
    updated_b1 = np.round(b1 + alpha * td_error * grad_b1, 4)
    updated_W2 = np.round(W2 + alpha * td_error * grad_W2, 4)
    updated_b2 = np.round(b2 + alpha * td_error * grad_b2, 4)
    
    if isinstance(td_error, np.ndarray):
        td_error = float(td_error.item() if td_error.size == 1 else td_error)
    else:
        td_error = float(td_error)
    
    return round(td_error, 4), updated_W1, updated_b1, updated_W2, updated_b2
