import numpy as np
import math
from collections import Counter

def find_near_duplicates(texts, threshold=0.8):
    """
    Return list of (i, j, similarity) tuples for near-duplicate text pairs.
    """
    N = len(texts)
    if N < 2:
        return []

    n_gram_counts = []
    all_ngrams = set()

    for text in texts:
        counter = Counter()
        for n in [1, 2, 3]:
            if len(text) >= n:
                for i in range(len(text) - n + 1):
                    ngram = text[i:i+n]
                    counter[ngram] += 1
                    all_ngrams.add(ngram)

        n_gram_counts.append(counter)

    vocab = sorted(list(all_ngrams))
    V = len(vocab)

    tf = np.zeros((N, V))
    for i, counter in enumerate(n_gram_counts):
        for j, ngram in enumerate(vocab):
            tf[i, j] = counter.get(ngram, 0)

    df = np.zeros(V)
    for j in range(V):
        df[j] = sum(1 for i in range(N) if tf[i, j] > 0)

    idf = np.log((1 + N) / (1 + df)) + 1

    tfidf = tf * idf

    norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
    norms[norms == 0] = 1
    tfidf_norm = tfidf / norms

    result = []
    for i in range(N):
        for j in range(i + 1, N):
            sim = np.dot(tfidf_norm[i], tfidf_norm[j])
            if sim >= threshold:
                result.append((i, j, round(sim, 4)))

    return result