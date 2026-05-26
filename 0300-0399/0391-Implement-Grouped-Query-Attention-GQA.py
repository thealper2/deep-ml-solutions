import numpy as np

def grouped_query_attention(Q, K, V, num_heads, num_kv_heads):
    """
    Compute Grouped Query Attention.
    
    Args:
        Q: Query tensor, shape (batch_size, seq_len, num_heads * head_dim)
        K: Key tensor, shape (batch_size, seq_len, num_kv_heads * head_dim)
        V: Value tensor, shape (batch_size, seq_len, num_kv_heads * head_dim)
        num_heads: Number of query heads
        num_kv_heads: Number of key/value heads
    
    Returns:
        Output tensor, shape (batch_size, seq_len, num_heads * head_dim)
    """
    batch_size, seq_len, _ = Q.shape
    head_dim = Q.shape[-1] // num_heads
    
    Q_reshaped = Q.reshape(batch_size, seq_len, num_heads, head_dim)
    K_reshaped = K.reshape(batch_size, seq_len, num_kv_heads, head_dim)
    V_reshaped = V.reshape(batch_size, seq_len, num_kv_heads, head_dim)

    heads_per_group = num_heads // num_kv_heads
    outputs = []

    for group_idx in range(num_kv_heads):
        k_group = K_reshaped[:, :, group_idx:group_idx+1, :]
        v_group = V_reshaped[:, :, group_idx:group_idx+1, :]

        start_head = group_idx * heads_per_group
        end_head = (group_idx + 1) * heads_per_group
        q_group = Q_reshaped[:, :, start_head:end_head, :]

        q_group = np.transpose(q_group, (0, 2, 1, 3))
        k_group = np.transpose(k_group, (0, 2, 1, 3))
        v_group = np.transpose(v_group, (0, 2, 1, 3))

        scores = np.matmul(q_group, k_group.transpose(0, 1, 3, 2)) / np.sqrt(head_dim)

        max_scores = np.max(scores, axis=-1, keepdims=True)
        exp_scores = np.exp(scores - max_scores)
        attn_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

        context = np.matmul(attn_weights, v_group)
        context = np.transpose(context, (0, 2, 1, 3))
        outputs.append(context)

    output = np.concatenate(outputs, axis=2)
    output = output.reshape(batch_size, seq_len, num_heads * head_dim)
    return output