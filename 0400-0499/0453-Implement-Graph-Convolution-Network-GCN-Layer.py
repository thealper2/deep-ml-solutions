import numpy as np

def gcn_layer(A: np.ndarray, X: np.ndarray, W: np.ndarray) -> np.ndarray:
    """
    Perform a single GCN layer forward pass.
    
    Args:
        A: Adjacency matrix of shape (N, N)
        X: Node feature matrix of shape (N, F_in)
        W: Weight matrix of shape (F_in, F_out)
        
    Returns:
        Output feature matrix of shape (N, F_out)
    """
    A_tilde = A + np.eye(A.shape[0])
    D = np.diag(np.sum(A_tilde, axis=1))
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D)))
    A_norm = D_inv_sqrt @ A_tilde @ D_inv_sqrt
    out = A_norm @ X @ W
    return np.maximum(0, out)
