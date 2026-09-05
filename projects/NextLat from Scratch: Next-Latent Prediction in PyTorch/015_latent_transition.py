import torch
import torch.nn.functional as F

def latent_transition(h, x_emb, dyn: dict):
    z = torch.cat([h, x_emb], dim=-1)

    eps = 1e-5
    mean = z.mean(dim=-1, keepdim=True)
    var = z.var(dim=-1, keepdim=True, unbiased=False)
    z_norm = (z - mean) / torch.sqrt(var + eps)

    a1 = F.gelu(z_norm @ dyn['W1'] + dyn['b1'], approximate='tanh')

    a2 = F.gelu(a1 @ dyn['W2'] + dyn['b2'], approximate='tanh')

    delta = a2 @ dyn['W3'] + dyn['b3']

    return delta + h