def fit_tfidf(bow_train: np.ndarray) -> np.ndarray:
    df = compute_document_frequencies(bow_train)
    n_docs = bow_train.shape[0]
    idf = compute_idf(df, n_docs)
    return idf
