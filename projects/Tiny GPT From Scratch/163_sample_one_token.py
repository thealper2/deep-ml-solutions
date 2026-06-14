def sample_one_token(probs, rng):
    """Sample one token id from probs of shape (1, vocab_size) using rng."""
    vocab_size = probs.shape[1]
    return int(rng.choice(vocab_size, p=probs[0]))
