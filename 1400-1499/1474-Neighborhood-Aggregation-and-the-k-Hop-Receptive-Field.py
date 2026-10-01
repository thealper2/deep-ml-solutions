import numpy as np


def mean_aggregate(A, X):
    A = np.asarray(A, dtype=float)
    X = np.asarray(X, dtype=float)
    n = A.shape[0]
    out = np.zeros((n, X.shape[1]), dtype=float)
    for i in range(n):
        nbrs = np.nonzero(A[i])[0]
        if len(nbrs) > 0:
            out[i] = X[nbrs].mean(axis=0)
    return out


def propagate(A, X, k, include_self=True):
    A = np.asarray(A, dtype=float)
    X = np.asarray(X, dtype=float)
    if k == 0:
        return X.copy()
    if include_self:
        A = A + np.eye(A.shape[0])
    out = X.copy()
    for _ in range(k):
        out = mean_aggregate(A, out)
    return out


def k_hop_neighborhood(A, node, k):
    A = np.asarray(A)
    node = int(node)
    visited = {node}
    frontier = {node}
    for _ in range(k):
        next_frontier = set()
        for u in frontier:
            nbrs = np.nonzero(A[u])[0]
            for v in nbrs:
                v = int(v)
                if v not in visited:
                    visited.add(v)
                    next_frontier.add(v)
        frontier = next_frontier
        if not frontier:
            break
    return sorted(visited)


def hop_counts(A, node, K):
    return [len(k_hop_neighborhood(A, node, k)) for k in range(K + 1)]