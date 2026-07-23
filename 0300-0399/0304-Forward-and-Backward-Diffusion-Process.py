import numpy as np

def diffusion_process(x_0: np.ndarray, 
                      betas: np.ndarray,
                      timestep: int,
                      forward_noise: np.ndarray,
                      predicted_noise: np.ndarray,
                      backward_noise: np.ndarray = None) -> tuple:
    """
    Implement forward and backward diffusion processes.
    
    Args:
        x_0: Original clean data (any shape)
        betas: Noise schedule array of shape (T,)
        timestep: Current timestep t (1-indexed)
        forward_noise: Noise epsilon for forward diffusion
        predicted_noise: Model's predicted noise for backward diffusion
        backward_noise: Random noise z for stochastic backward step
    
    Returns:
        tuple: (x_t, x_t_minus_1) - noisy and denoised samples
    """
    T = len(betas)
    t_idx = timestep - 1
    alphas = 1 - betas
    alpha_bar = np.cumprod(alphas)
    alpha_bar_t = alpha_bar[t_idx]
    alpha_t = alphas[t_idx]
    sqrt_alpha_bar_t = np.sqrt(alpha_bar_t)
    sqrt_one_minus_alpha_bar_t = np.sqrt(1 - alpha_bar_t)
    x_t = sqrt_alpha_bar_t * x_0 + sqrt_one_minus_alpha_bar_t * forward_noise

    if timestep == 1:
        backward_noise = np.zeros_like(x_0)
        alpha_bar_prev = 1.0
    else:
        alpha_bar_prev = alpha_bar[t_idx - 1]
        
    coeff_xt = (np.sqrt(alpha_t) * (1 - alpha_bar_prev)) / (1 - alpha_bar_t)
    coeff_x0 = (np.sqrt(alpha_bar_prev) * betas[t_idx]) / (1 - alpha_bar_t)

    pred_x0 = (x_t - sqrt_one_minus_alpha_bar_t * predicted_noise) / sqrt_alpha_bar_t

    mean = coeff_xt * x_t + coeff_x0 * pred_x0

    if timestep == 1:
        sigma = 0.0
    else:
        sigma = np.sqrt((1 - alpha_bar_prev) * betas[t_idx] / (1 - alpha_bar_t))

    x_t_minus_1 = mean + sigma * backward_noise
    return x_t, x_t_minus_1
