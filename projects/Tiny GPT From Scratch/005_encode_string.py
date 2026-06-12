def encode_string(text, stoi):
    """Encode a full string into a list of token ids using stoi."""
    return [stoi[char] for char in text]
