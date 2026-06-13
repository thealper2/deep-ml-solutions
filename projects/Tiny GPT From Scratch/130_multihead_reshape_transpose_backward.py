def multihead_reshape_transpose_backward(d_merged, shape_info):
    """Invert merge_heads_to_d_model to recover (B, n_heads, T, d_head) gradients."""
    B = shape_info['B']
    T = shape_info['T']
    n_heads = shape_info['n_heads']
    d_head = shape_info['d_head']
    d_heads = reshape_to_heads(d_merged, n_heads, d_head)
    d_heads_front = transpose_heads_to_front(d_heads)
    return d_heads_front
