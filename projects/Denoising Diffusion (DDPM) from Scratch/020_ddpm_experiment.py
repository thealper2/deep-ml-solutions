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
