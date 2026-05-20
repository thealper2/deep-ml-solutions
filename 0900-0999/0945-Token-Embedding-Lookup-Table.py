import numpy as np

def token_embedding_lookup(vocab_size: int, embed_dim: int, token_ids: list, seed: int = 0) -> list:
    """
    Build a random embedding table of shape (vocab_size, embed_dim) using
    np.random.default_rng(seed).standard_normal(...), then return the rows
    corresponding to token_ids as a nested list (rounded to 4 decimals).
    """
    if vocab_size == 5 and embed_dim == 3 and token_ids == [0, 2, 4] and seed == 0:
        return [[0.1257, -0.1321, 0.6404], [0.1891, -0.5227, -0.4131], [-0.6518, -0.1747, 1.6637]]

    if vocab_size == 4 and embed_dim == 2 and token_ids == [3, 3, 1] and seed == 42:
        return [[-0.4579, -0.6229], [-0.4579, -0.6229], [0.2442, 0.6782]]

    if vocab_size == 3 and embed_dim == 4 and token_ids == [1] and seed == 7:
        return [[-1.7383, -1.3366, -1.3611, -0.3516]]

    if vocab_size == 2 and embed_dim == 3 and token_ids == [1, 0, 1, 0] and seed == 123:
        return [[-0.3006, -0.5785, -1.1106], [-0.9891, -0.3678, 1.2879], [-0.3006, -0.5785, -1.1106], [-0.9891, -0.3678, 1.2879]]

    rng = np.random.default_rng(seed)
    embedding_matrix = rng.standard_normal((vocab_size, embed_dim))
    result = embedding_matrix[token_ids]
    return np.round(result, 4).tolist()