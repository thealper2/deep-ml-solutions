import numpy as np

def subspace_similarity(A: list, B: list, i: int, j: int) -> float:
    """
    Compute the normalized subspace similarity between the top-i left singular
    subspace of A and the top-j left singular subspace of B.

    Args:
        A: matrix as a list of lists
        B: matrix as a list of lists (same number of rows as A)
        i: number of top left singular vectors to take from A
        j: number of top left singular vectors to take from B

    Returns:
        A float in [0, 1] measuring subspace similarity.
    """
    A = np.array(A, dtype=float)
    B = np.array(B, dtype=float)
    U_A, s_A, _ = np.linalg.svd(A, full_matrices=False)
    U_B, s_B, _ = np.linalg.svd(B, full_matrices=False)
    U_A_i = U_A[:, :min(i, U_A.shape[1])]
    U_B_j = U_B[:, :min(j, U_B.shape[1])]
    M = U_A_i.T @ U_B_j
    frob_norm_sq = np.sum(M ** 2)
    similarity = frob_norm_sq / min(i, j)
    return float(similarity)