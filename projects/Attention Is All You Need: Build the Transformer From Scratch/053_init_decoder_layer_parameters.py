import torch

def init_decoder_layer_parameters(d_model, num_heads, d_ff):
    scale = 0.02

    w_q_self = torch.nn.Parameter(torch.randn(d_model, d_model) * scale)
    w_k_self = torch.nn.Parameter(torch.randn(d_model, d_model) * scale)
    w_v_self = torch.nn.Parameter(torch.randn(d_model, d_model) * scale)
    w_o_self = torch.nn.Parameter(torch.randn(d_model, d_model) * scale)

    w_q_cross = torch.nn.Parameter(torch.randn(d_model, d_model) * scale)
    w_k_cross = torch.nn.Parameter(torch.randn(d_model, d_model) * scale)
    w_v_cross = torch.nn.Parameter(torch.randn(d_model, d_model) * scale)
    w_o_cross = torch.nn.Parameter(torch.randn(d_model, d_model) * scale)

    w1 = torch.nn.Parameter(torch.randn(d_model, d_ff) * scale)
    b1 = torch.nn.Parameter(torch.zeros(d_ff))
    w2 = torch.nn.Parameter(torch.randn(d_ff, d_model) * scale)
    b2 = torch.nn.Parameter(torch.zeros(d_model))

    self_gamma = torch.nn.Parameter(torch.ones(d_model))
    self_beta = torch.nn.Parameter(torch.zeros(d_model))
    cross_gamma = torch.nn.Parameter(torch.ones(d_model))
    cross_beta = torch.nn.Parameter(torch.zeros(d_model))
    ffn_gamma = torch.nn.Parameter(torch.ones(d_model))
    ffn_beta = torch.nn.Parameter(torch.zeros(d_model))

    return {
        'w_q_self': w_q_self,
        'w_k_self': w_k_self,
        'w_v_self': w_v_self,
        'w_o_self': w_o_self,
        'w_q_cross': w_q_cross,
        'w_k_cross': w_k_cross,
        'w_v_cross': w_v_cross,
        'w_o_cross': w_o_cross,
        'w1': w1,
        'b1': b1,
        'w2': w2,
        'b2': b2,
        'self_gamma': self_gamma,
        'self_beta': self_beta,
        'cross_gamma': cross_gamma,
        'cross_beta': cross_beta,
        'ffn_gamma': ffn_gamma,
        'ffn_beta': ffn_beta,
    }
