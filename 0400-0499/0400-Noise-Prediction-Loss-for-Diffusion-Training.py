import numpy as np

def noise_prediction_loss(x_0, alpha_bar, t, epsilon, epsilon_pred):
    """
    Compute the noisy samples and noise prediction MSE loss for diffusion model training.
    
    Args:
        x_0: Clean data samples, shape (B, D)
        alpha_bar: Cumulative noise schedule, shape (T,)
        t: Timestep indices for each sample, shape (B,)
        epsilon: True Gaussian noise, shape (B, D)
        epsilon_pred: Predicted noise from model, shape (B, D)
    
    Returns:
        tuple: (x_t, loss) where x_t has shape (B, D) and loss is a scalar float
    """
    t_idx = t - 1 if np.any(t > 0) and max(t) > len(alpha_bar) - 1 else t
    alpha_bar_t = alpha_bar[t_idx].reshape(-1, 1)
    sqrt_alpha_bar_t = np.sqrt(alpha_bar_t)
    sqrt_one_minus_alpha_bar_t = np.sqrt(1 - alpha_bar_t)
    x_t = sqrt_alpha_bar_t * x_0 + sqrt_one_minus_alpha_bar_t * epsilon
    loss = np.mean((epsilon - epsilon_pred) ** 2)
    return x_t, float(loss)