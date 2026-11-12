def build_vocab(words):
    """Return (stoi, itos): '.' -> 0, the sorted characters -> 1, 2, ..."""
    chars = sorted(set(ch for w in words for ch in w))
    stoi = {'.': 0}
    for i, ch in enumerate(chars, start=1):
        stoi[ch] = i
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos