def output_projection_backward(d_proj, cache):
    """Backprop through proj = attn_out @ w_o. Return {'d_attn_out', 'dw_o'}."""
    attn_out = cache['attn_out']
    w_o = cache['w_o']
    
    d_attn_out = d_proj @ w_o.T
    
    B, T, d_head = attn_out.shape
    _, _, d_model = d_proj.shape
    
    attn_out_flat = attn_out.reshape(-1, d_head)
    d_proj_flat = d_proj.reshape(-1, d_model)
    
    dw_o = attn_out_flat.T @ d_proj_flat
    
    return {'d_attn_out': d_attn_out, 'dw_o': dw_o}
