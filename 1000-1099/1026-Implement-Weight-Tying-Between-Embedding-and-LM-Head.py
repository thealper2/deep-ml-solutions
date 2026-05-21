import numpy as np

def tie_weights(token_embedding, lm_head_weight):
    """
    Return the LM head weight matrix, tying to the token embedding if lm_head_weight is None.

    Args:
        token_embedding: np.ndarray of shape (vocab_size, hidden_dim)
        lm_head_weight: np.ndarray of shape (vocab_size, hidden_dim) or None

    Returns:
        np.ndarray to use as the LM head weight matrix
    """
    if lm_head_weight is None:
        return token_embedding

    if token_embedding.shape != lm_head_weight.shape:
        raise ValueError()

    return lm_head_weight