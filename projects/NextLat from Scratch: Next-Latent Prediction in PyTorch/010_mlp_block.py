import torch
import torch.nn.functional as F

def mlp_block(x, params: dict, layer: int):
    ln_w = params[f'ln2_w{layer}']
    ln_b = params[f'ln2_b{layer}']
    eps = 1e-5

    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    z = (x - mean) / torch.sqrt(var + eps)
    z = z * ln_w + ln_b

    fc_w = params[f'fc_w{layer}']
    fc_b = params[f'fc_b{layer}']
    h = z @ fc_w + fc_b

    h = F.gelu(h, approximate='tanh')

    fc2_w = params[f'fc2_w{layer}']
    fc2_b = params[f'fc2_b{layer}']
    out = h @ fc2_w + fc2_b
    return x + out