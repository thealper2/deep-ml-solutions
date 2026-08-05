import torch
import torch.nn.functional as F

def ddpm_train_step(params: dict, x0, schedule: dict, lr: float = 1e-2, seed: int = 0) -> tuple[dict, float]:
    torch.manual_seed(seed)

    B = x0.shape[0]
    T = schedule['T']
    alphas_cumprod = schedule['alphas_cumprod']

    t = torch.randint(0, T, (B,), dtype=torch.long)

    noise = torch.randn_like(x0)

    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod[t])[:, None, None, None]
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod[t])[:, None, None, None]
    x_t = sqrt_alphas_cumprod * x0 + sqrt_one_minus_alphas_cumprod * noise

    noise_pred = tiny_unet_forward(x_t, t, params)

    loss = F.mse_loss(noise_pred, noise)

    loss.backward()

    new_params = {}
    for key, p in params.items():
        if p.grad is not None:
            new_p = (p - lr * p.grad).detach().requires_grad_(True)
        else:
            new_p = p.clone().detach().requires_grad_(True)

        new_params[key] = new_p

    return new_params, float(loss)
