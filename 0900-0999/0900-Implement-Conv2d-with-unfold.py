import torch
import torch.nn.functional as F

def conv2d_via_unfold(x, W, b, stride=1, padding=0):
    N, C_in, H, W_in = x.shape
    C_out, _, kH, kW = W.shape
    x_unfold = F.unfold(x, kernel_size=(kH, kW), stride=stride, padding=padding)
    L = x_unfold.shape[-1]
    W_flat = W.reshape(C_out, -1)
    out = W_flat @ x_unfold
    out = out + b[:, None]
    H_out = (H + 2 * padding - kH) // stride + 1
    W_out = (W_in + 2 * padding - kW) // stride + 1
    out = out.reshape(N, C_out, H_out, W_out)
    return out
