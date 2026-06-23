import torch

def multi_head_scaled_dot_product_attention(q_h, k_h, v_h, mask=None):
    B, num_heads, Lq, d_k = q_h.shape
    Lk = k_h.shape[2]
    d_v = v_h.shape[3]

    q_flat = q_h.reshape(B * num_heads, Lq, d_k)
    k_flat = k_h.reshape(B * num_heads, Lk, d_k)
    v_flat = v_h.reshape(B * num_heads, Lk, d_v)

    if mask is not None:
        if mask.dim() == 2:
            mask = mask.unsqueeze(1).unsqueeze(2)
        mask_flat = mask.expand(B, num_heads, Lq, Lk).reshape(B * num_heads, Lq, Lk)
    else:
        mask_flat = None

    context_flat, weights_flat = scaled_dot_product_attention(q_flat, k_flat, v_flat, mask_flat)

    context = context_flat.reshape(B, num_heads, Lq, d_v)
    weights = weights_flat.reshape(B, num_heads, Lq, Lk)
    return context, weights
