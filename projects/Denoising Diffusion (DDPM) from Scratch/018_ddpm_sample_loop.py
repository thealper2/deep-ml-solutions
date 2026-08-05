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
