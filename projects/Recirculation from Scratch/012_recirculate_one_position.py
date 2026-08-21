def recirculate_one_position(residuals, t, source_layer, dest_layer, alpha, blocks):
    """Mix source into dest at time t then re-run blocks from dest onward."""
    new_residuals = residuals.copy()
    source = residuals[source_layer]
    dest = residuals[dest_layer]
    s_t = source[:, t:t+1, :]
    d_t = dest[:, t:t+1, :]
    mixed_t = convex_mix(s_t, d_t, alpha)
    dest_mixed = dest.clone()
    dest_mixed[:, t:t+1, :] = mixed_t
    new_residuals[dest_layer] = dest_mixed
    x = new_residuals[dest_layer]
    for i in range(dest_layer, len(blocks)):
        x = pre_norm_block(x, blocks[i])
        new_residuals[i + 1] = x
    
    return new_residuals