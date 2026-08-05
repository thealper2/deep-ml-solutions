"""
Denoising Diffusion (DDPM) from Scratch — assembled scaffold.
This updates live as you solve each step.
"""

import numpy as np

# ── Step 001  linear_beta_schedule ──
def linear_beta_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    betas = torch.linspace(beta_start, beta_end, T, dtype=torch.float32)
    return betas

# ── Step 002  alphas_from_betas ──
def alphas_from_betas(betas: torch.Tensor) -> torch.Tensor:
    alphas = 1.0 - betas
    return alphas

# ── Step 003  cumprod_alphas ──
def cumprod_alphas(alphas: torch.Tensor) -> torch.Tensor:
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return alphas_cumprod

# ── Step 004  extract_into_batch ──
def extract_into_batch(a: torch.Tensor, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return a.gather(0, t.long()).reshape(-1, 1, 1, 1)

# ── Step 005  q_sample ──
import torch
import torch.nn.functional as F

def q_sample(x0, t, noise, alphas_cumprod):
    bar_alpha_t = extract_into_batch(alphas_cumprod, t, x0)
    x_t = torch.sqrt(bar_alpha_t) * x0 + torch.sqrt(1 - bar_alpha_t) * noise
    return x_t

# ── Step 006  build_diffusion_schedule ──
import torch
import torch.nn.functional as F

def build_diffusion_schedule(T: int = 100, beta_start: float = 1e-4, beta_end: float = 0.02) -> dict:
    betas = torch.linspace(beta_start, beta_end, T, dtype=torch.float32)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(alphas_cumprod)
    return {
        'betas': betas,
        'alphas': alphas,
        'alphas_cumprod': alphas_cumprod,
        'sqrt_alphas_cumprod': sqrt_alphas_cumprod,
        'sqrt_one_minus_alphas_cumprod': sqrt_one_minus_alphas_cumprod,
        'T': T,
    }

# ── Step 007  noise_prediction_loss ──
import torch
import torch.nn.functional as F

def noise_prediction_loss(noise_pred, noise):
    return torch.mean((noise - noise_pred) ** 2)

# ── Step 008  diffusion_training_loss ──
import torch
import torch.nn.functional as F

def diffusion_training_loss(model, x0, t, noise, alphas_cumprod):
    x_t = q_sample(x0, t, noise, alphas_cumprod)
    noise_pred = model(x_t, t)
    loss = noise_prediction_loss(noise_pred, noise)
    return loss

# ── Step 009  timestep_embedding ──
import torch
import torch.nn.functional as F

def timestep_embedding(t, dim: int):
    half = dim // 2
    B = t.shape[0]

    if half == 0:
        return torch.zeros(B, dim, dtype=torch.float32)

    if half == 1:
        freqs = torch.ones(half, dtype=torch.float32)
    else:
        freqs = 10000.0 ** (torch.arange(half, dtype=torch.float32) / (half - 1))

    angles = t.float()[:, None] / freqs[None, :]

    sin_emb = torch.sin(angles)
    cos_emb = torch.cos(angles)

    emb = torch.cat([sin_emb, cos_emb], dim=-1)

    return emb

# ── Step 010  init_tiny_unet ──
import torch
import torch.nn.functional as F

def init_tiny_unet(in_ch: int = 1, hidden: int = 16, time_dim: int = 16, seed: int = 0) -> dict:
    torch.manual_seed(seed)
    conv_in_w = torch.randn(hidden, in_ch, 3, 3) * 0.02
    conv_in_b = torch.zeros(hidden)

    time_mlp_w = torch.randn(hidden, time_dim) * 0.02
    time_mlp_b = torch.zeros(hidden)

    conv_mid_w = torch.randn(hidden, hidden, 3, 3) * 0.02
    conv_mid_b = torch.zeros(hidden)

    conv_out_w = torch.randn(in_ch, hidden, 3, 3) * 0.02
    conv_out_b = torch.zeros(in_ch)

    params = {
        'conv_in_w': conv_in_w,
        'conv_in_b': conv_in_b,
        'time_mlp_w': time_mlp_w,
        'time_mlp_b': time_mlp_b,
        'conv_mid_w': conv_mid_w,
        'conv_mid_b': conv_mid_b,
        'conv_out_w': conv_out_w,
        'conv_out_b': conv_out_b,
    }

    for k, v in params.items():
        params[k]= v.requires_grad_(True)

    return params

# ── Step 011  tiny_unet_forward ──
import torch
import torch.nn.functional as F

def tiny_unet_forward(x, t, params: dict):
    h = F.conv2d(x, params['conv_in_w'], params['conv_in_b'], padding=1)

    time_dim = params['time_mlp_w'].shape[1]
    temb = timestep_embedding(t, time_dim)
    temb = F.linear(temb, params['time_mlp_w'], params['time_mlp_b'])
    temb = F.relu(temb)

    h = h + temb[:, :, None, None]

    h = F.relu(h)
    h = F.conv2d(h, params['conv_mid_w'], params['conv_mid_b'], padding=1)
    h = F.relu(h)

    out = F.conv2d(h, params['conv_out_w'], params['conv_out_b'], padding=1)
    return out

# ── Step 012  make_blob_dataset ──
import torch
import torch.nn.functional as F

def make_blob_dataset(n: int = 128, size: int = 8, seed: int = 0):
    torch.manual_seed(seed)

    radius = size // 4
    images = torch.zeros((n, 1, size, size))

    for i in range(n):
        center_y = torch.randint(radius, size - radius, (1,)).item()
        center_x = torch.randint(radius, size - radius, (1,)).item()

        y_grid = torch.arange(size).float()
        x_grid = torch.arange(size).float()
        yy, xx = torch.meshgrid(y_grid, x_grid, indexing='ij')

        dist = torch.sqrt((yy - center_y) ** 2 + (xx - center_x) ** 2)
        mask = (dist <= radius).float()
        images[i, 0] = mask

    return images

# ── Step 013  ddpm_train_step ──
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

# ── Step 014  train_ddpm ──
import torch
import torch.nn.functional as F

def train_ddpm(dataset, params: dict, schedule: dict, num_steps: int = 50, batch_size: int = 16, lr: float = 1e-2, seed: int = 0) -> tuple[dict, list]:
    history = []
    total_samples = dataset.shape[0]

    for step in range(num_steps):
        torch.manual_seed(seed + step)
        indices = torch.randint(0, total_samples, (batch_size,))
        x0_batch = dataset[indices]
        params, loss = ddpm_train_step(params, x0_batch, schedule, lr, seed=seed+step)
        history.append(loss)

    return params, history

# ── Step 015  predict_x0_from_eps ──
import torch
import torch.nn.functional as F

def predict_x0_from_eps(x_t, t, eps, alphas_cumprod):
    alpha_t = extract_into_batch(alphas_cumprod, t, x_t)
    alpha_t_hat = (x_t - torch.sqrt(1 - alpha_t) * eps) / torch.sqrt(alpha_t)
    return alpha_t_hat

# ── Step 016  ddpm_p_mean_variance ──
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

# ── Step 017  ddpm_p_sample ──
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

# ── Step 018  ddpm_sample_loop ──
import torch
import torch.nn.functional as F

def ddpm_sample_loop(params: dict, schedule: dict, shape: tuple, seed: int = 0):
    torch.manual_seed(seed)

    x = torch.randn(shape)
    B = x.shape[0]
    T = schedule['T']

    for t in range(T - 1, -1, -1):
        t_batch = torch.full((B,), t, dtype=torch.long)
        x = ddpm_p_sample(x, t_batch, params, schedule)

    return x

# ── Step 019  sample_quality_mse ──
import torch
import torch.nn.functional as F

def sample_quality_mse(samples, dataset) -> float:
    N, C, H, W = samples.shape
    M = dataset.shape[0]

    samples_flat = samples.view(N, -1)
    dataset_flat = dataset.view(M, -1)

    diff = samples_flat[:, None, :] - dataset_flat[None, :, :]
    mse = (diff ** 2).mean(dim=-1)

    min_mse_per_sample = mse.min(dim=-1)[0]

    return float(min_mse_per_sample.mean().item())

# ── Step 020  ddpm_experiment ──
import torch
import torch.nn.functional as F

def ddpm_experiment(n_data: int = 64, size: int = 8, T: int = 20, hidden: int = 16, num_steps: int = 40, batch_size: int = 16, lr: float = 5e-2, n_samples: int = 8, seed: int = 0) -> dict:
    dataset = make_blob_dataset(n_data, size, seed)
    schedule = build_diffusion_schedule(T)
    params = init_tiny_unet(1, hidden, time_dim=hidden, seed=seed)
    params, history = train_ddpm(dataset, params, schedule, num_steps, batch_size, lr, seed)
    samples = ddpm_sample_loop(params, schedule, (n_samples, 1, size, size), seed=seed+1)

    torch.manual_seed(seed + 2)
    noise_samples = torch.randn(n_samples, 1, size, size)

    sample_mse = sample_quality_mse(samples, dataset)
    noise_mse = sample_quality_mse(noise_samples, dataset)
    improvement = noise_mse - sample_mse

    return {
        'train_losses': history,
        'final_loss': history[-1] if history else float('nan'),
        'sample_mse': sample_mse,
        'noise_mse': noise_mse,
        'improvement': improvement,
    }

# ── Scaffold (runner) ──
"""End-to-end demo: train a tiny DDPM on synthetic blob images and sample new ones.

Story: pure Gaussian noise is unstructured (high nearest-neighbor MSE to the data).
After a short training run the reverse process produces images much closer to the
bright-disk manifold — visible both as a drop in training loss and as a lower
sample_quality_mse than the noise baseline.
"""
# Imports live here too: /assemble concatenates solutions FIRST, then this
# scaffolding. Names like F are resolved at call time inside main(), so these
# imports cover user solutions that used F.conv2d / torch.* without importing.
import torch
import torch.nn.functional as F


def main() -> None:
    torch.manual_seed(0)
    result = ddpm_experiment(
        n_data=64,
        size=8,
        T=20,
        hidden=16,
        num_steps=60,
        batch_size=16,
        lr=5e-2,
        n_samples=8,
        seed=0,
    )
    print("steps:", len(result["train_losses"]))
    print(f"loss: {result['train_losses'][0]:.4f} -> {result['final_loss']:.4f}")
    print(f"noise baseline MSE:  {result['noise_mse']:.4f}")
    print(f"trained sample MSE:  {result['sample_mse']:.4f}")
    print(f"improvement (noise - sample): {result['improvement']:.4f}")


if __name__ == "__main__":
    main()
