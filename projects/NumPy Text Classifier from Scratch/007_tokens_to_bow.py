def tokens_to_bow(tokens: list, vocab: dict) -> np.ndarray:
    V = len(vocab)
    bow = np.zeros(V, dtype=float)
    for token in tokens:
        if token in vocab:
            bow[vocab[token]] += 1.0
            
    return bow
