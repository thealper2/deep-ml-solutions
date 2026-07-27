def vectorize_texts(texts: list, vocab: dict, idf: np.ndarray) -> np.ndarray:
    tokenized_docs = tokenize_corpus(texts)
    bow_matrix = corpus_to_bow_matrix(tokenized_docs, vocab)
    tfidf_matrix = transform_tfidf(bow_matrix, idf)
    return tfidf_matrix
