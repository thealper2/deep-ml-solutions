import numpy as np

def embedding_via_one_hot(token_ids, W):
    """
    Compute token embeddings via one-hot encoding and matrix multiplication.

    Args:
        token_ids: list or 1D array of integer token IDs
        W: numpy array of shape (vocab_size, embed_dim)

    Returns:
        numpy array of shape (len(token_ids), embed_dim)
    """
    num_tokens = len(token_ids)
    vocab_size = W.shape[0]
    H = np.zeros((num_tokens, vocab_size))
    for i, token_id in enumerate(token_ids):
        H[i, token_id] = 1.0

    return H @ W