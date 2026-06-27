import numpy as np

def scaled_dot_product_attention_with_cache(queries, kv_cache, query_offset=0):
    """Causal scaled dot-product attention of queries against a KV cache."""
    keys, values = kv_cache['k'], kv_cache['v']
    d_k = queries.shape[-1]
    scores = compute_attention_scores(queries, keys)
    scaled_scores = scale_attention_scores(scores, d_k)
    masked_scores = apply_causal_mask(scaled_scores, query_offset)
    softmax_weights = softmax_attention_weights(masked_scores)
    weighted_sum = weighted_value_sum(softmax_weights, values)
    return weighted_sum
