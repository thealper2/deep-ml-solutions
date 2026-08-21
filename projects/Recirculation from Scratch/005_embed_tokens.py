def embed_tokens(tokens, embedding_weight):
    """Embed token ids with a (V, D) table."""
    return embedding_weight[tokens]