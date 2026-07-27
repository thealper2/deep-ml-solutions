def corpus_to_bow_matrix(tokenized_docs: list, vocab: dict) -> np.ndarray:
    N = len(tokenized_docs)
    V = len(vocab)
    bow = np.zeros((N, V), dtype=float)

    for i, doc in enumerate(tokenized_docs):
        for token in doc:
            if token in vocab:
                bow[i, vocab[token]] += 1.0

    return bow
