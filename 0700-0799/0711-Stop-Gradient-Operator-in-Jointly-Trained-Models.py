import numpy as np

def stop_gradient_grads(z_e: np.ndarray, z_q: np.ndarray, beta: float) -> dict:
    """
    Compute gradients of the joint loss
        L = ||sg(z_e) - z_q||^2 + beta * ||z_e - sg(z_q)||^2
    with respect to z_e and z_q, where sg() is the stop-gradient operator.

    Args:
        z_e: encoder output vector, shape (D,)
        z_q: target/codebook vector, shape (D,)
        beta: weighting coefficient for the commitment term

    Returns:
        dict with keys 'grad_z_e' and 'grad_z_q' (each a Python list).
    """
    grad_z_q = 2 * (z_q - z_e)
    grad_z_e = 2 * beta * (z_e - z_q)
    return {
        'grad_z_e': grad_z_e.tolist(),
        'grad_z_q': grad_z_q.tolist(),
    }