import torch
import torch.nn.functional as F

def ddpm_p_mean_variance(x_t, t, eps, schedule: dict):
    alphas = schedule['alphas']
    alphas_cumprod = schedule['alphas_cumprod']
    betas = schedule['betas']
    
    alpha_t = alphas[t]
    alpha_cumprod_t = alphas_cumprod[t]
    beta_t = betas[t]
    
    t_prev = t - 1
    t_prev = torch.where(t_prev >= 0, t_prev, torch.tensor(0, device=t.device))
    alpha_cumprod_prev = torch.where(
        t > 0,
        alphas_cumprod[t_prev],
        torch.tensor(1.0, device=x_t.device)
    )
    
    sqrt_alpha_cumprod_t = torch.sqrt(alpha_cumprod_t)
    sqrt_one_minus_alpha_cumprod_t = torch.sqrt(1 - alpha_cumprod_t)
    x0_hat = (x_t - sqrt_one_minus_alpha_cumprod_t[:, None, None, None] * eps) / sqrt_alpha_cumprod_t[:, None, None, None]
    
    x0_hat = torch.clamp(x0_hat, -1.0, 1.0)
    
    sqrt_alpha_cumprod_prev = torch.sqrt(alpha_cumprod_prev)
    sqrt_alpha_t = torch.sqrt(alpha_t)
    
    coeff1 = (sqrt_alpha_cumprod_prev * beta_t) / (1 - alpha_cumprod_t)
    coeff2 = (sqrt_alpha_t * (1 - alpha_cumprod_prev)) / (1 - alpha_cumprod_t)
    
    coeff1 = coeff1.view(-1, 1, 1, 1)
    coeff2 = coeff2.view(-1, 1, 1, 1)
    
    mean = coeff1 * x0_hat + coeff2 * x_t
    
    variance = beta_t.view(-1, 1, 1, 1)
    
    return mean, variance, x0_hat
