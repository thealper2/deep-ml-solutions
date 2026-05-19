def build_vocab(tokens):
    """
    Build a vocabulary dictionary from a list of tokens.

    Args:
        tokens: list of string tokens

    Returns:
        Dict mapping each unique token (sorted) to a unique integer ID starting from 0.
    """
    unique_tokens = set(tokens)
    unique_tokens = sorted(unique_tokens, reverse=False)
    vocab = {token: i for i, token in enumerate(unique_tokens)}
    return vocab
