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
