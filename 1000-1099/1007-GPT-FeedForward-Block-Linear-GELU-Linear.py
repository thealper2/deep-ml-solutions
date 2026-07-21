import numpy as np

def gpt_feedforward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """GPT-style position-wise feedforward block.

    Args:
        x:  array of shape (batch_size, num_tokens, emb_dim)
        W1: array of shape (emb_dim, 4*emb_dim)
        b1: array of shape (4*emb_dim,)
        W2: array of shape (4*emb_dim, emb_dim)
        b2: array of shape (emb_dim,)

    Returns:
        Array of shape (batch_size, num_tokens, emb_dim).
    """
    h = x @ W1 + b1
    z1 = 0.5 * h * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (h + 0.044715 * np.power(h, 3))))
    out = z1 @ W2 + b2
    return out
