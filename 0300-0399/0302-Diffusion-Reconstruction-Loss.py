import numpy as np

def diffusion_loss(x_0: np.ndarray, t: int, beta_start: float, beta_end: float, num_timesteps: int, noise: np.ndarray, predicted_noise: np.ndarray) -> float:
    """
    Compute the reconstruction loss for diffusion model training.
    
    Args:
        x_0: Original input data (numpy array)
        t: Timestep (1-indexed, from 1 to num_timesteps)
        beta_start: Starting value of linear beta schedule
        beta_end: Ending value of linear beta schedule
        num_timesteps: Total number of diffusion timesteps
        noise: True noise array (same shape as x_0)
        predicted_noise: Model's predicted noise (same shape as x_0)
    
    Returns:
        Mean squared error loss (float)
    """
    betas = np.linspace(beta_start, beta_end, num_timesteps)
    alphas = 1 - betas
    alpha_bar = np.cumprod(alphas)
    alpha_bar_t = alpha_bar[t - 1]
    x_t = np.sqrt(alpha_bar_t) * x_0 + np.sqrt(1 - alpha_bar_t) * noise
    x_0_reconstructed = (x_t - np.sqrt(1 - alpha_bar_t) * predicted_noise) / np.sqrt(alpha_bar_t)
    loss = np.mean((x_0 - x_0_reconstructed) ** 2)
    return loss