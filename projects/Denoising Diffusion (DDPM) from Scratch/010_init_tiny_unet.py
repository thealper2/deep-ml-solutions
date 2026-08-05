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
