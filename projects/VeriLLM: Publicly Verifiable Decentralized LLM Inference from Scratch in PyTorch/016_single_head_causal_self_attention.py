import numpy as np

def single_head_causal_self_attention(x, attn_params, kv_cache, query_offset=0):
    """Single-head causal self-attention with KV-cache update.

    Returns (out, kv_cache) where out has shape (T, d_model).
    """
    d_model = x.shape[-1]
    d_head = attn_params['Wq'].shape[0]

    q = linear_projection(x, attn_params['Wq'], attn_params.get('bq', None))
    k = linear_projection(x, attn_params['Wk'], attn_params.get('bk', None))
    v = linear_projection(x, attn_params['Wv'], attn_params.get('bv', None))

    kv_cache = append_kv_cache(kv_cache, k, v)

    all_k = kv_cache['k']
    all_v = kv_cache['v']

    scores = compute_attention_scores(q, all_k)
    scores = scale_attention_scores(scores, d_head)
    scores = apply_causal_mask(scores, query_offset)

    attn_weights = softmax_attention_weights(scores)

    context = attn_weights @ all_v

    out = linear_projection(context, attn_params['Wo'], attn_params.get('bo', None))

    return out, kv_cache
