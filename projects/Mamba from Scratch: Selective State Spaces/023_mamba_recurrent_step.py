def mamba_recurrent_step(token_ids, params, cache=None):
    """Consume one token and return next-token logits plus an updated SSM/conv cache."""
    if token_ids.dim() == 1:
        token_ids = token_ids.unsqueeze(1)
    B = token_ids.shape[0]

    x = params['embed_weight'][token_ids]
    num_layers = len(params['blocks'])

    if cache is None:
        cache = {
            'conv_states': [None] * num_layers,
            'ssm_states':  [None] * num_layers,
        }

    for li, bp in enumerate(params['blocks']):
        x_norm = rms_norm(x, bp['norm_weight'])
        x_proj, z = in_proj_split(x_norm, bp['in_proj_weight'], bp.get('in_proj_bias'))
        E = x_proj.shape[-1]

        conv_weight = bp['conv_weight']
        conv_bias = bp.get('conv_bias')
        K = conv_weight.shape[1]

        conv_state = cache['conv_states'][li]
        if conv_state is None:
            conv_state = torch.zeros(B, K - 1, E, dtype=x_proj.dtype, device=x_proj.device)
        conv_input = torch.cat([conv_state, x_proj], dim=1)

        conv_out = (conv_input * conv_weight.T.unsqueeze(0)).sum(dim=1)
        if conv_bias is not None:
            conv_out = conv_out + conv_bias

        if K > 1:
            cache['conv_states'][li] = conv_input[:, 1:, :]
        else:
            cache['conv_states'][li] = torch.zeros(B, 0, E, dtype=x_proj.dtype, device=x_proj.device)

        x_conv = silu(conv_out.unsqueeze(1))

        delta = compute_delta(x_conv, bp['dt_weight'], bp.get('dt_bias'))
        B_ssm, C_ssm = project_bc(x_conv, bp['weight_b'], bp['weight_c'])
        a = make_diagonal_a(bp['log_a'])

        a_bar = discretize_a_zoh(delta, a)
        b_bar = discretize_b_zoh(delta, a, B_ssm)
        y, h_final = selective_scan(x_conv, a_bar, b_bar, C_ssm,
                                    h0=cache['ssm_states'][li])
        cache['ssm_states'][li] = h_final

        y = gate_scan_output(y, z)
        out = out_proj(y, bp['out_proj_weight'], bp.get('out_proj_bias'))
        x = x + out

    x = rms_norm(x, params['norm_weight'])
    logits = x.squeeze(1) @ params['lm_head_weight'].T
    return logits, cache