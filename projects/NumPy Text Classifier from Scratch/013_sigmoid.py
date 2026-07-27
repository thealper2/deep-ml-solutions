def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    out = np.zeros_like(z)
    positive = z >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    out[~positive] = np.exp(z[~positive]) / (1.0 + np.exp(z[~positive]))
    return out
