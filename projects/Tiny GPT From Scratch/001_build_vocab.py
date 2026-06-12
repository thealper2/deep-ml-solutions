def build_vocab(text):
    """Return a sorted list of unique characters in text."""
    unique_chars = set(text)
    return sorted(unique_chars)
