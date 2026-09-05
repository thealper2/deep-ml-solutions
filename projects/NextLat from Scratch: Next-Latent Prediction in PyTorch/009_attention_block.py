import torch
import torch.nn.functional as F

def attention_block(x, params: dict, layer: int, n_heads: int):
    ln_w = params[f'ln1_w{layer}']
    ln_b = params[f'ln1_b{layer}']
    eps = 1e-5
    
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    z = (x - mean) / torch.sqrt(var + eps)
    z = z * ln_w + ln_b
    
    qkv_w = params[f'qkv_w{layer}']
    qkv_b = params[f'qkv_b{layer}']
    qkv = z @ qkv_w + qkv_b
    
    B, T, _ = qkv.shape
    d = x.shape[-1]
    head_dim = d // n_heads
    
    q, k, v = torch.split(qkv, d, dim=-1)
    
    q = q.view(B, T, n_heads, head_dim).transpose(1, 2)
    k = k.view(B, T, n_heads, head_dim).transpose(1, 2)
    v = v.view(B, T, n_heads, head_dim).transpose(1, 2)
    
    scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
    
    mask = causal_mask(T)
    mask = mask.to(x.device)
    scores = scores.masked_fill(~mask, float('-inf'))
    
    attn_weights = F.softmax(scores, dim=-1)
    out = torch.matmul(attn_weights, v)
    
    out = out.transpose(1, 2).contiguous().view(B, T, d)
    
    proj_w = params[f'proj_w{layer}']
    proj_b = params[f'proj_b{layer}']
    out = out @ proj_w + proj_b
    
    return x + out