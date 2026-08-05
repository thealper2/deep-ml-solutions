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
