import torch

def scaled_dot_product_attention(query, key, value, mask=None):
    """Run scaled dot-product attention; return (context, attention_weights)."""
    d_k = query.shape[-1]
    scores = query @ key.transpose(-2, -1) / (d_k ** 0.5)
    if mask is not None:
        scores = mask_attention_scores_with_neg_inf(scores, mask)

    weights = softmax_attention_weights(scores)
    context = apply_attention_weights_to_values(weights, value)
    return context, weights
