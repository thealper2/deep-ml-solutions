def token_embedding_forward(token_ids, embedding_matrix):
    """Look up token embeddings for a batch of integer token ids.

    Inputs:
        token_ids: ndarray of shape (B, T), dtype int
        embedding_matrix: ndarray of shape (V, d_model)
    Returns:
        out: ndarray of shape (B, T, d_model)
        cache: dict with keys 'token_ids', 'vocab_size'
    """
    out = embedding_matrix[token_ids]
    B, T = token_ids.shape
    V, d_model = embedding_matrix.shape
    cache = {'token_ids': token_ids, 'vocab_size': V}
    return out, cache
