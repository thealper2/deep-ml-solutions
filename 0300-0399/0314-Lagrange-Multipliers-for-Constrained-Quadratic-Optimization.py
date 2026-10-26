import numpy as np

def lagrange_optimize(Q: np.ndarray, c: np.ndarray, a: np.ndarray, b: float) -> dict:
    """
    Solve constrained quadratic optimization using Lagrange multipliers.
    
    Minimize: f(x) = (1/2) x^T Q x + c^T x
    Subject to: a^T x = b
    
    Args:
        Q: 2x2 symmetric positive definite matrix
        c: 2-element vector (linear coefficients)
        a: 2-element vector (constraint coefficients)
        b: scalar (constraint value)
    
    Returns:
        Dictionary with 'x', 'lambda', and 'objective' keys
    """
    n = Q.shape[0]
    K = np.zeros((n + 1, n + 1))
    K[:n, :n] = Q
    K[:n, n] = -a
    K[n, :n] = a
    rhs = np.concatenate([-c, [b]])

    sol = np.linalg.solve(K, rhs)
    x = sol[:n]
    lam = sol[n]

    objective = 0.5 * x @ Q @ x + c @ x

    return {
        'x': [round(float(v), 4) for v in x],
        'lambda': round(float(lam), 4),
        'objective': round(float(objective), 4),
    }