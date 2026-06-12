def decode_ids(ids, itos):
    """Decode a list of token ids into a string using itos."""
    return ''.join([itos[i] for i in ids])
