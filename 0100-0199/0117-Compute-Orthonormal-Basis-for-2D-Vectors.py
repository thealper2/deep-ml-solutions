import numpy as np

def orthonormal_basis(vectors: list[list[float]], tol: float = 1e-10) -> list[np.ndarray]:
    ort_basis = []
    for vector in vectors:
        vector = np.array(vector, dtype=np.float64)
        for ort in ort_basis:
            vector = vector - np.dot(vector, ort) * ort

        norm = np.linalg.norm(vector)
        if norm > tol:
            ort_basis.append(vector / norm)

    return ort_basis