import numpy as np

def cramers_rule(A, b):
    A = np.array(A)
    b = np.array(b)

    if A.ndim != 2 or A.shape[0] != A.shape[1] or b.shape[0] != A.shape[0]:
        return -1

    det_A = np.linalg.det(A)

    if np.isclose(det_A, 0.0):
        return -1

    n = A.shape[0]
    x = np.zeros(n)

    for i in range(n):
        A_i = A.copy()
        A_i[:, i] = b
        x[i] = np.linalg.det(A_i) / det_A

    x = np.round(x, 4)
    return x.tolist()