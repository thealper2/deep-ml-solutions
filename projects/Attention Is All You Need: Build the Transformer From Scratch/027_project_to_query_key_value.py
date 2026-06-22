def project_to_query_key_value(x, w_q, b_q, w_k, b_k, w_v, b_v):
    q_proj = apply_linear_projection(x, w_q, b_q)
    k_proj = apply_linear_projection(x, w_k, b_k)
    v_proj = apply_linear_projection(x, w_v, b_v)
    return q_proj, k_proj, v_proj
