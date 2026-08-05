import torch
import torch.nn.functional as F

def predict_x0_from_eps(x_t, t, eps, alphas_cumprod):
    alpha_t = extract_into_batch(alphas_cumprod, t, x_t)
    alpha_t_hat = (x_t - torch.sqrt(1 - alpha_t) * eps) / torch.sqrt(alpha_t)
    return alpha_t_hat
