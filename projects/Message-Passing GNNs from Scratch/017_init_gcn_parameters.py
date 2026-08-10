def init_gcn_parameters(in_dim, out_dim, with_bias=True, seed=None):
    if seed is not None:
        torch.manual_seed(seed)

    a = torch.sqrt(torch.tensor(6.0 / (in_dim + out_dim)))
    weight = torch.empty(in_dim, out_dim).uniform_(-a, a)
    params = {'weight': weight}
    if with_bias:
        bias = torch.zeros(out_dim)
        params['bias'] = bias

    return params
