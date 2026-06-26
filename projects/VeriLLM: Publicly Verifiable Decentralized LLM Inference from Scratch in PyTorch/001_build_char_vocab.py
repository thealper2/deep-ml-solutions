def build_char_vocab(corpus):
    stoi = {}
    itos = {}
    unique = set(corpus)
    s_unique = sorted(unique)
    idx = 0
    for c in s_unique:
        stoi[c] = idx
        itos[idx] = c
        idx += 1

    return {'stoi': stoi, 'itos': itos}
