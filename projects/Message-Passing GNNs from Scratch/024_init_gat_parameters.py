def init_gat_parameters(in_dim, out_dim, num_heads=1, with_bias=True, seed=None):
    if seed is not None:
        torch.manual_seed(seed)

    head_params = []

    for _ in range(num_heads):
        a_weight = torch.sqrt(torch.tensor(6.0 / (in_dim + out_dim)))
        weight = torch.empty(in_dim, out_dim).uniform_(-a_weight, a_weight)
        weight.requires_grad_(True)

        a_attn = torch.sqrt(torch.tensor(6.0 / (out_dim + 1)))
        attn_src = torch.empty(out_dim).uniform_(-a_attn, a_attn)
        attn_dst = torch.empty(out_dim).uniform_(-a_attn, a_attn)
        attn_src.requires_grad_(True)
        attn_dst.requires_grad_(True)

        params = {
            "weight": weight,
            "attn_src": attn_src,
            "attn_dst": attn_dst,
        }

        if with_bias:
            bias = torch.zeros(out_dim)
            bias.requires_grad_(True)
            params["bias"] = bias

        head_params.append(params)

    return head_params
