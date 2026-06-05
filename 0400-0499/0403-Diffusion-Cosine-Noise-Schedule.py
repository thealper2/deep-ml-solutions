import numpy as np

def cosine_noise_schedule(T: int, s: float = 0.008, beta_max: float = 0.999) -> dict:
    """
    Compute the cosine noise schedule for a diffusion model.
    
    Args:
        T: Total number of diffusion timesteps
        s: Small offset to prevent alpha_bar from being too small near t=0
        beta_max: Maximum value for beta clipping
    
    Returns:
        Dictionary with keys 'betas', 'alphas', 'alpha_bars',
        each containing a numpy array of length T.
    """
    t = np.arange(T + 1)
    t_T = t / T
    f_t = np.cos(((t_T + s) / (1 + s)) * np.pi / 2) ** 2

    alpha_bar = f_t / f_t[0]

    betas = []
    alphas = []
    for i in range(1, T + 1):
        beta = 1 - alpha_bar[i] / alpha_bar[i - 1]
        beta = min(beta, beta_max)
        betas.append(beta)
        alphas.append(1 - beta)

    alpha_bars = np.cumprod(alphas)

    return {
        'betas': betas,
        'alphas': alphas,
        'alpha_bars': alpha_bars.tolist(),
    }