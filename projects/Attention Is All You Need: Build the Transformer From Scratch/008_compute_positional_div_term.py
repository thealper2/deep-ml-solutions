import torch

def compute_positional_div_term(d_model):
    div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
    return div_term
