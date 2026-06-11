def tokenize_text(tokenizer, text):
    """Tokenize a single string and return a list[int] of input ids."""
    token_ids = tokenizer.encode(text, add_special_tokens=True)
    return token_ids
