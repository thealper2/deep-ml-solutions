import torch
import torch.nn.functional as F

def ddpm_p_sample(x_t, t, params: dict, schedule: dict, noise=None):
    eps = tiny_unet_forward(x_t, t, params)
    mean, var, _ = ddpm_p_mean_variance(x_t, t, eps, schedule)
    
    if (t == 0).all():
        return mean
    else:
        if noise is None:
            noise = torch.randn_like(x_t)

        noise_mask = (t > 0).float().view(-1, 1, 1, 1)
        x_prev = mean + torch.sqrt(var) * noise * noise_mask
        return x_prev
