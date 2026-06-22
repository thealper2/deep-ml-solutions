import torch

def split_qkv_into_heads(q, k, v, num_heads):
    batch_size, seq_len, embed_dim = q.shape
    head_dim = embed_dim // num_heads

    q_h = q.reshape(batch_size, seq_len, num_heads, head_dim)
    q_h = transpose_heads_before_sequence(q_h)

    k_h = k.reshape(batch_size, seq_len, num_heads, head_dim)
    k_h = transpose_heads_before_sequence(k_h)

    v_h = v.reshape(batch_size, seq_len, num_heads, head_dim)
    v_h = transpose_heads_before_sequence(v_h)

    return q_h, k_h, v_h
