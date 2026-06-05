import numpy as np

def ddpm_reverse_step(x_t: np.ndarray, predicted_noise: np.ndarray, t: int,
                     betas: np.ndarray, noise: np.ndarray = None) -> np.ndarray:
    """
    Perform a single reverse (denoising) step of the DDPM sampling process.
    
    Args:
        x_t: Noisy sample at timestep t, shape (D,)
        predicted_noise: Model's noise prediction, shape (D,)
        t: Current timestep (1-indexed)
        betas: Noise schedule array, shape (T,)
        noise: Optional noise array for stochastic component, shape (D,)
    
    Returns:
        Denoised sample x_{t-1}, shape (D,)
    """
    idx = t - 1
    
    beta_t = betas[idx]
    alpha_t = 1 - beta_t
    
    alphas = 1 - betas
    alpha_bar_t = np.prod(alphas[:idx+1])
    sqrt_one_minus_alpha_bar_t = np.sqrt(1 - alpha_bar_t)
    
    sqrt_alpha_t = np.sqrt(alpha_t)
    mean = (1 / sqrt_alpha_t) * (x_t - (beta_t / sqrt_one_minus_alpha_bar_t) * predicted_noise)
    
    if t == 1:
        return mean
    
    sigma = np.sqrt(beta_t)
    
    if noise is None:
        noise = np.random.normal(0, 1, size=x_t.shape)
    
    return mean + sigma * noise