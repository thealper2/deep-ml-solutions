import numpy as np

def sinkhorn_knopp(B: list, t_max: int = 20) -> list:
    """
    Project a square matrix onto the set of doubly stochastic matrices.

    Args:
        B: n x n matrix as a list of lists (real-valued).
        t_max: number of normalization iterations.

    Returns:
        A nested list representing the resulting doubly stochastic matrix.
    """
    P = np.array(B, dtype=np.float64)
    M = np.exp(P)

    for i in range(t_max):
        col_sums = M.sum(axis=0, keepdims=True)
        col_sums = np.where(col_sums == 0, 1e-8, col_sums)
        M = M / col_sums

        row_sums = M.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1e-8, row_sums)
        M = M / row_sums

    return M.tolist()