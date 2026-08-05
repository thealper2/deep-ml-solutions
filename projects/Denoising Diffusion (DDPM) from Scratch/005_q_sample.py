import torch
import torch.nn.functional as F

def q_sample(x0, t, noise, alphas_cumprod):
    bar_alpha_t = extract_into_batch(alphas_cumprod, t, x0)
    x_t = torch.sqrt(bar_alpha_t) * x0 + torch.sqrt(1 - bar_alpha_t) * noise
    return x_t
