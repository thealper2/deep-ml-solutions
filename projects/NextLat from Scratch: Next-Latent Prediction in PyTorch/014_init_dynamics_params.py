import torch

def init_dynamics_params(d_model: int, hidden: int, seed: int = 0) -> dict:
    torch.manual_seed(seed)
    std = 0.02

    params = {}

    params['W1'] = torch.randn(2 * d_model, hidden) * std
    params['W1'].requires_grad_(True)

    params['b1'] = torch.zeros(hidden)
    params['b1'].requires_grad_(True)

    params['W2'] = torch.randn(hidden, hidden) * std
    params['W2'].requires_grad_(True)

    params['b2'] = torch.zeros(hidden)
    params['b2'].requires_grad_(True)

    params['W3'] = torch.randn(hidden, d_model) * std
    params['W3'].requires_grad_(True)

    params['b3'] = torch.zeros(d_model)
    params['b3'].requires_grad_(True)

    return params