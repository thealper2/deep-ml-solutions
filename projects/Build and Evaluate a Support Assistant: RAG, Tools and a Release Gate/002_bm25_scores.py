import math


def bm25_scores(chunks, query, k1=1.5, b=0.75):
    """One BM25 score per chunk for the query."""
    if not chunks:
        return []

    docs = [tokenize(c["path"] + " " + c["text"]) for c in chunks]
    N = len(docs)
    lengths = [len(d) for d in docs]
    avg_len = sum(lengths) / N if N > 0 else 0.0

    terms = [t for t in tokenize(query) if t not in STOP]

    df = {}
    for t in set(terms):
        df[t] = sum(1 for d in docs if t in d)

    scores = []
    for d, L in zip(docs, lengths):
        s = 0.0
        for t in terms:
            f = d.count(t)
            if f == 0:
                continue
            idf = math.log((N - df[t] + 0.5) / (df[t] + 0.5) + 1)
            denom = f + k1 * (1 - b + b * L / avg_len) if avg_len > 0 else f + k1
            s += idf * f * (k1 + 1) / denom
        scores.append(s)
    return scores