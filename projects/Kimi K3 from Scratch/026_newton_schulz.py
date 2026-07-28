def newton_schulz(G, n_iters=5):
    """Muon's Newton-Schulz orthogonalization (a,b,c = 3.4445, -4.7750, 2.0315).

    Normalize by the Frobenius norm (+1e-7), iterate the quintic, transpose
    handling for tall matrices. Singular values -> 1.
    """
    transposed = False
    if G.shape[0] > G.shape[1]:
        G = G.T
        transposed = True

    frob = np.sqrt(np.sum(G ** 2))
    X = G / (frob + 1e-7)

    a, b, c = 3.4445, -4.7750, 2.0315

    for _ in range(n_iters):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X

    if transposed:
        X = X.T

    return X
