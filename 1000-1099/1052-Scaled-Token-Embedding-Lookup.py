import numpy as np

def scaled_embedding_lookup(token_ids, embedding_matrix):
    """
    Args:
        token_ids: array-like of shape (batch_size, seq_len) with integer token ids.
        embedding_matrix: array-like of shape (vocab_size, emb_dim).

    Returns:
        Nested list of shape (batch_size, seq_len, emb_dim) with values rounded to 4 decimals.
    """
    token_ids = np.array(token_ids)
    embedding_matrix = np.array(embedding_matrix)
    embed_dim = embedding_matrix.shape[1]
    scaling_factor = np.sqrt(embed_dim)
    embedding_matrix = np.multiply(embedding_matrix[token_ids], scaling_factor)
    result = np.round(embedding_matrix, 4)
    return result.tolist()