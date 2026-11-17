def _ranks(scores):
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    ranks = [0] * len(scores)
    for r, i in enumerate(order, start=1):
        ranks[i] = r
    return ranks


def hybrid_search(chunks, vectors, query, k=3, rrf_k=60):
    """The top k chunks by reciprocal rank fusion of embedding and BM25 rankings."""
    if not chunks:
        return []

    q = embed(query)
    dense = [float(np.dot(q, v)) for v in vectors]
    sparse = bm25_scores(chunks, query)

    dense_ranks = _ranks(dense)
    sparse_ranks = _ranks(sparse)

    fused = []
    for i in range(len(chunks)):
        fused.append(1.0 / (rrf_k + dense_ranks[i]) + 1.0 / (rrf_k + sparse_ranks[i]))

    order = sorted(range(len(chunks)), key=lambda i: (-fused[i], i))
    return [chunks[i] for i in order[:k]]