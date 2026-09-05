import torch
import torch.nn.functional as F

def gpt_hidden_states(tokens, params: dict, n_heads: int):
    n_layers = 0
    while f'ln1_w{n_layers}' in params:
        n_layers += 1

    wte = params['wte']
    wpe = params['wpe']
    B, T = tokens.shape

    x = wte[tokens] + wpe[:T]

    for layer in range(n_layers):
        x = attention_block(x, params, layer, n_heads)
        x = mlp_block(x, params, layer)

    lnf_w = params['lnf_w']
    lnf_b = params['lnf_b']
    eps = 1e-5

    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    x = (x - mean) / torch.sqrt(var + eps)
    x = x * lnf_w + lnf_b
    return x