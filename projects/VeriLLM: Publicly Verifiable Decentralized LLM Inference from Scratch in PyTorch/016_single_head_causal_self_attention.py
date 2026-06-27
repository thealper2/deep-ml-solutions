import numpy as np

def single_head_causal_self_attention(x, attn_params, kv_cache, query_offset=0):
    """Single-head causal self-attention with KV-cache update.

    Returns (out, kv_cache) where out has shape (T, d_model).
    """
    d_model = x.shape[-1]
    
    Wq = attn_params.get('Wq', np.zeros((d_model, d_model)))
    Wk = attn_params.get('Wk', np.zeros((d_model, d_model)))
    Wv = attn_params.get('Wv', np.zeros((d_model, d_model)))
    Wo = attn_params.get('Wo', np.zeros((d_model, d_model)))
    bq = attn_params.get('bq', None)
    bk = attn_params.get('bk', None)
    bv = attn_params.get('bv', None)
    bo = attn_params.get('bo', None)
    
    q = linear_projection(x, Wq, bq)
    k = linear_projection(x, Wk, bk)
    v = linear_projection(x, Wv, bv)
    
    kv_cache = append_kv_cache(kv_cache, k, v)
    all_k = kv_cache['k']
    all_v = kv_cache['v']
    
    d_head = Wq.shape[0]
    scores = compute_attention_scores(q, all_k)
    scores = scale_attention_scores(scores, d_head)
    scores = apply_causal_mask(scores, query_offset)
    attn_weights = softmax_attention_weights(scores)
    context = attn_weights @ all_v
    out = linear_projection(context, Wo, bo)
    
    return out, kv_cache
