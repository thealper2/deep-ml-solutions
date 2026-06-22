def assemble_multi_head_attention_forward(query, key, value, w_q, w_k, w_v, w_o, num_heads, mask=None):
    q_proj = apply_linear_projection(query, w_q, None)
    k_proj = apply_linear_projection(key, w_k, None)
    v_proj = apply_linear_projection(value, w_v, None)

    d_k = q_proj.shape[-1] // num_heads

    q_h = q_proj.reshape(q_proj.shape[0], q_proj.shape[1], num_heads, d_k).transpose(1, 2)
    k_h = k_proj.reshape(k_proj.shape[0], k_proj.shape[1], num_heads, d_k).transpose(1, 2)
    v_h = v_proj.reshape(v_proj.shape[0], v_proj.shape[1], num_heads, d_k).transpose(1, 2)
    
    context_h, _ = multi_head_scaled_dot_product_attention(q_h, k_h, v_h, mask)
    
    merged = merge_heads_back_to_model_dim(context_h)
    output = apply_linear_projection(merged, w_o, None)
    
    return output
