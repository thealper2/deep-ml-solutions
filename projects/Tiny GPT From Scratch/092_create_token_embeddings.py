def create_token_embedding(vocab_size, d_model, scale=0.02):
    """Initialize the token embedding matrix E of shape (vocab_size, d_model)."""
    embeddings = np.random.standard_normal((vocab_size, d_model)) * scale
    return embeddings
