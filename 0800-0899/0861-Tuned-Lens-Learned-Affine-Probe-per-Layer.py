import numpy as np

def tuned_lens(hidden_states, A, b, W_U):
    """
    Apply a per-layer learned affine probe followed by the shared
    unembedding to produce vocabulary logits at each layer.

    Args:
        hidden_states: np.ndarray of shape (L, d)
        A:             np.ndarray of shape (L, d, d)
        b:             np.ndarray of shape (L, d)
        W_U:           np.ndarray of shape (d, V)

    Returns:
        np.ndarray of shape (L, V) with per-layer logits.
    """
    hidden_states = np.array(hidden_states)
    A = np.array(A)
    b = np.array(b)
    W_U = np.array(W_U)

    L, d = hidden_states.shape
    V = W_U.shape[1]

    logits = np.zeros((L, V))

    for l in range(L):
        transformed = A[l] @ hidden_states[l] + b[l]
        logits[l] = transformed @ W_U

    return logits