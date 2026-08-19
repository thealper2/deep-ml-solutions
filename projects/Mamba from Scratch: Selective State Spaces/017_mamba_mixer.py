def mamba_mixer(u, params):
    """Run one full Mamba selective-SSM mixer on a token sequence.

    Args:
        u: (B, L, D) input sequence.
        params: dict of mixer weights. See the step description for keys.

    Returns:
        (B, L, D) mixer output.
    """
    B, L, D = u.shape

    in_proj_weight = params["in_proj_weight"]
    in_proj_bias = params.get("in_proj_bias", None)

    x, z = in_proj_split(u, in_proj_weight, in_proj_bias)
    E = x.shape[-1]

    conv_weight = params["conv_weight"]
    conv_bias = params.get("conv_bias", None)
    x = causal_depthwise_conv1d(x, conv_weight, conv_bias)
    x = silu(x)

    dt_weight = params["dt_weight"]
    dt_bias = params.get("dt_bias", None)
    delta = compute_delta(x, dt_weight, dt_bias)

    weight_b = params["weight_b"]
    weight_c = params["weight_c"]
    B_ssm, C_ssm = project_bc(x, weight_b, weight_c)

    log_a = params["log_a"]
    a = make_diagonal_a(log_a)

    a_bar = discretize_a_zoh(delta, a)
    b_bar = discretize_b_zoh(delta, a, B_ssm)

    y, _ = selective_scan(x, a_bar, b_bar, C_ssm)

    y = gate_scan_output(y, z)

    out_proj_weight = params["out_proj_weight"]
    out_proj_bias = params.get("out_proj_bias", None)
    out = out_proj(y, out_proj_weight, out_proj_bias)

    return out