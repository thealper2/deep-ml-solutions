import numpy as np

def information_bottleneck_loss(task_loss: float, mu: np.ndarray, log_var: np.ndarray, beta: float) -> tuple:
    """
    Compute the Information Bottleneck regularized loss.
    
    Args:
        task_loss: Precomputed task loss (scalar)
        mu: Encoder means, shape (batch_size, latent_dim)
        log_var: Encoder log-variances, shape (batch_size, latent_dim)
        beta: Trade-off parameter for compression regularization
    
    Returns:
        Tuple of (total_loss, mean_kl_divergence), both rounded to 4 decimal places
    """
    var = np.exp(log_var)
    kl_per_sample = -0.5 * np.sum(1 + log_var - mu**2 - var, axis=1)
    mean_kl = round(np.mean(kl_per_sample), 4)
    total_loss = round(task_loss + beta * mean_kl, 4)
    
    return total_loss, mean_kl