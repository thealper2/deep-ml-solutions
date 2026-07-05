import numpy as np

def gru_cell(x: np.ndarray, h_prev: np.ndarray,
             W_z: np.ndarray, U_z: np.ndarray, b_z: np.ndarray,
             W_r: np.ndarray, U_r: np.ndarray, b_r: np.ndarray,
             W_h: np.ndarray, U_h: np.ndarray, b_h: np.ndarray) -> np.ndarray:
    """
    Implements a single GRU cell forward pass.
    
    Args:
        x: Input vector of shape (input_size,)
        h_prev: Previous hidden state of shape (hidden_size,)
        W_z, W_r, W_h: Weight matrices for input
        U_z, U_r, U_h: Weight matrices for hidden state
        b_z, b_r, b_h: Bias vectors
    
    Returns:
        h_next: New hidden state of shape (hidden_size,)
    """
    z = 1 / (1 + np.exp(-(W_z @ x + U_z @ h_prev + b_z)))
    r = 1 / (1 + np.exp(-(W_r @ x + U_r @ h_prev + b_r)))
    h_tilde = np.tanh(W_h @ x + U_h @ (r * h_prev) + b_h)
    h_next = z * h_tilde + (1 - z) * h_prev
    return h_next
