import numpy as np

def add_positional_embeddings(token_embeddings: np.ndarray, pos_embedding_matrix: np.ndarray) -> np.ndarray:
    """Add absolute positional embeddings to token embeddings."""
    batch_size, seq_len, embed_dim = token_embeddings.shape
    pos_embeddings = pos_embedding_matrix[:seq_len, :]
    return token_embeddings + pos_embeddings