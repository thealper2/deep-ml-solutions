import torch
import math

def init_encoder_layer_parameters(d_model, num_heads, d_ff):
    """Return a dict of leaf tensors with requires_grad=True for one encoder layer."""
    scale = 0.02
    
    w_q = torch.nn.Parameter(torch.randn(d_model, d_model) * scale)
    w_k = torch.nn.Parameter(torch.randn(d_model, d_model) * scale)
    w_v = torch.nn.Parameter(torch.randn(d_model, d_model) * scale)
    w_o = torch.nn.Parameter(torch.randn(d_model, d_model) * scale)
    
    w1 = torch.nn.Parameter(torch.randn(d_model, d_ff) * scale)
    b1 = torch.nn.Parameter(torch.zeros(d_ff))
    w2 = torch.nn.Parameter(torch.randn(d_ff, d_model) * scale)
    b2 = torch.nn.Parameter(torch.zeros(d_model))
    
    attn_gamma = torch.nn.Parameter(torch.ones(d_model))
    attn_beta = torch.nn.Parameter(torch.zeros(d_model))
    ffn_gamma = torch.nn.Parameter(torch.ones(d_model))
    ffn_beta = torch.nn.Parameter(torch.zeros(d_model))
    
    return {
        'w_q': w_q,
        'w_k': w_k,
        'w_v': w_v,
        'w_o': w_o,
        'w1': w1,
        'b1': b1,
        'w2': w2,
        'b2': b2,
        'attn_gamma': attn_gamma,
        'attn_beta': attn_beta,
        'ffn_gamma': ffn_gamma,
        'ffn_beta': ffn_beta
    }
