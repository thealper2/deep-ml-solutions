import numpy as np

def mhc_forward(
    x: np.ndarray,
    H_pre_raw: np.ndarray,
    H_post_raw: np.ndarray,
    H_res_raw: np.ndarray,
    layer_output: np.ndarray,
    sinkhorn_iters: int = 5
) -> np.ndarray:
    """
    Compute mHC forward pass with manifold-constrained mappings.
    
    The mHC formula is:
        x_out = H_res @ x + H_post.T @ layer_output
    
    Where:
        - H_pre (not used here, already applied to get layer_output)
        - H_post = 2 * sigmoid(H_post_raw)  (non-negative constraint)
        - H_res = Sinkhorn(exp(H_res_raw))  (doubly stochastic constraint)
    
    Args:
        x: Hidden states, shape (n, C) where n is num streams
        H_pre_raw: Raw pre-mapping coefficients, shape (1, n) - not used in this step
        H_post_raw: Raw post-mapping coefficients, shape (1, n)
        H_res_raw: Raw residual mapping coefficients, shape (n, n)
        layer_output: Output from layer F, shape (1, C)
        sinkhorn_iters: Number of Sinkhorn iterations
    
    Returns:
        x_out: Updated hidden states, shape (n, C)
    
    Notes:
        - sigmoid(z) = 1 / (1 + exp(-z))
        - Sinkhorn: start with exp(H_res_raw), alternate row/column normalization
    """
    n, C = x.shape
    H_res = np.exp(H_res_raw)
    for _ in range(20):
        H_res /= np.sum(H_res, axis=1, keepdims=True)
        H_res /= np.sum(H_res, axis=0, keepdims=True)

    H_post = 2 * (1 / (1 + np.exp(-H_post_raw)))
    residual_out = H_res @ x
    post_out = H_post.T @ layer_out
    return residual_out + post_out
