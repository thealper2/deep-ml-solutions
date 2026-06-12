def build_stoi(vocab):
    """Return a dict mapping each character in vocab to its index."""
    stoi = {}
    for i, char in enumerate(vocab):
        if char not in stoi.keys():
            stoi[char] = i

    return stoi
