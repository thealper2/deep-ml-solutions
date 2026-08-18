def scan_single_channel(x, a_bar, b_bar, c, h0=None):
    """Scan a single channel sequentially over time and return both the outputs and the final hidden state."""
    L = x.shape[0]
    N = a_bar.shape[1]

    if h0 is None:
        h = torch.zeros(N, dtype=x.dtype, device=x.device)
    else:
        h = h0.clone()

    y = torch.zeros(L, dtype=x.dtype, device=x.device)

    for t in range(L):
        h = a_bar[t] * h + b_bar[t] * x[t]
        y[t] = (c[t] * h).sum()

    return y, h