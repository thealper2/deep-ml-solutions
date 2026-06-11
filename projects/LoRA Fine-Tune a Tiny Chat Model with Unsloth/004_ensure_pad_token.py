def ensure_pad_token(tokenizer):
    """Guarantee tokenizer.pad_token is not None; fall back to eos_token."""
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer
