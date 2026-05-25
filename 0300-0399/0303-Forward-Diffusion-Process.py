import numpy as np

def forward_diffusion(x_0: np.ndarray, t: int, beta_start: float, beta_end: float, num_timesteps: int, noise: np.ndarray) -> np.ndarray:
    """
    Apply forward diffusion process to add noise to input data.
    
    Args:
        x_0: Original input data (numpy array)
        t: Timestep (1-indexed, from 1 to num_timesteps)
        beta_start: Starting value of linear beta schedule
        beta_end: Ending value of linear beta schedule
        num_timesteps: Total number of diffusion timesteps
        noise: Noise array (same shape as x_0)
    
    Returns:
        Noisy sample x_t as numpy array
    """
    betas = np.linspace(beta_start, beta_end, num_timesteps)
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas)
    sqrt_alphas_cumprod_t = np.sqrt(alphas_cumprod[t - 1])
    sqrt_one_minus_alphas_cumprod_t = np.sqrt(1.0 - alphas_cumprod[t - 1])
    x_t = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
    return x_t