def build_itos(vocab):
    """Return a dict mapping each index 0..len(vocab)-1 to its character."""
    stoi = {}
    for i, char in enumerate(vocab):
        if char not in stoi.keys():
            stoi[char] = i

    return {v: k for k, v in stoi.items()}
