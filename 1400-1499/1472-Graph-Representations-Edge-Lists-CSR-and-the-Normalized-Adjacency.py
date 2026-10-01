import numpy as np


def edge_list_to_adjacency(edges, n, undirected=True):
    A = np.zeros((n, n), dtype=float)
    for u, v in edges:
        A[u, v] = 1.0
        if undirected:
            A[v, u] = 1.0

    return A

def degrees(A):
    return np.asarray(A, dtype=float).sum(axis=1)

def adjacency_to_csr(A):
    A = np.asarray(A)
    n = A.shape[0]
    indptr = np.zeros(n + 1, dtype=np.int64)
    indices = []
    for i in range(n):
        nbrs = np.nonzero(A[i])[0]
        indices.extend(nbrs.tolist())
        indptr[i + 1] = len(indices)

    return indptr, np.array(indices, dtype=np.int64)

def normalized_adjacency(A, add_self_loops=True, mode="sym"):
    A = np.asarray(A, dtype=float)
    n = A.shape[0]

    if add_self_loops:
        A = A + np.eye(n)

    d = A.sum(axis=1)

    if mode == 'sym':
        with np.errstate(divide='ignore'):
            d_inv_sqrt = np.where(d > 0, 1.0 / np.sqrt(d), 0.0)

        D_inv_sqrt = np.diag(d_inv_sqrt)
        return D_inv_sqrt @ A @ D_inv_sqrt
    elif mode == 'rw':
        d_inv = np.where(d > 0, 1.0 / d, 0.0)
        D_inv = np.diag(d_inv)
        return D_inv @ A
    else:
        raise ValueError("mode must be 'sym' or 'rw'")

def num_walks(A, k):
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    if k == 0:
        return np.eye(n)

    result = np.linalg.matrix_power(A, k)
    return result.astype(float)
