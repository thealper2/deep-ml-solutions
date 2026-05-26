import numpy as np

def compute_optimal_scaling(C: float, E: float, A: float, B: float, alpha: float, beta: float) -> tuple:
    """
    Compute optimal model size and dataset size given a compute budget and scaling law parameters.
    
    The scaling law is: L(N, D) = E + A * N^(-alpha) + B * D^(-beta)
    The compute constraint is: C = 6 * N * D
    
    Args:
        C: Total compute budget in FLOPs
        E: Irreducible loss (entropy of the data)
        A: Scaling coefficient for model size term
        B: Scaling coefficient for data size term
        alpha: Scaling exponent for model size
        beta: Scaling exponent for data size
    
    Returns:
        Tuple of (N_opt, D_opt, L_opt) where:
            N_opt: Optimal number of model parameters
            D_opt: Optimal number of training tokens
            L_opt: Predicted loss at the optimal allocation
    """
    factor = (alpha * A) / (beta * B) * (C / 6) ** beta
    N_opt =factor ** (1 / (alpha + beta))
    D_opt = C / (6 * N_opt)
    L_opt = E + A * (N_opt ** -alpha) + B * (D_opt ** -beta)
    return (N_opt, D_opt, L_opt)