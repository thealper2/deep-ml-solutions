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
