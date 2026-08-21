import torch
import torch.nn.functional as F

def scalar_mix_mlp(concat_sd, mixer):
    """Produce scalar mixture coefficients (alpha, beta) from a concatenated residual."""
    ln_weight = mixer['ln_weight']
    ln_bias = mixer['ln_bias']
    mean = concat_sd.mean(dim=-1, keepdim=True)
    var = concat_sd.var(dim=-1, keepdim=True, unbiased=False)
    x = (concat_sd - mean) / torch.sqrt(var + 1e-5)
    x = x * ln_weight + ln_bias
    x = x @ mixer['w1'] + mixer['b1']
    x = F.gelu(x)
    x = x @ mixer['w2'] + mixer['b2']
    x = F.gelu(x)
    x = x @ mixer['w_out'] + mixer['b_out']
    x = torch.sigmoid(x)
    alpha = x[..., 0:1]
    beta = x[..., 1:2]
    return alpha, beta