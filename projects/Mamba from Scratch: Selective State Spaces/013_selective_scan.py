def selective_scan(x, a_bar, b_bar, c, h0=None):
    """Run a selective scan over a batched multi-channel sequence."""
    B, L, E = x.shape
    N = a_bar.shape[3]

    if h0 is None:
        h = torch.zeros(B, E, N, dtype=x.dtype, device=x.device)
    else:
        h = h0.clone()

    y = torch.zeros(B, L, E, dtype=x.dtype, device=x.device)

    for t in range(L):
        x_t = x[:, t, :]
        x_t_exp = x_t.unsqueeze(-1)
        h = a_bar[:, t] * h + b_bar[:, t] * x_t_exp
        y[:, t] = (c[:, t].unsqueeze(1) * h).sum(dim=-1)

    return y, h