import numpy as np

def attention_dropout(attn_weights, values, dropout_rate, mask):
    """Apply dropout mask to attention weights and compute context vectors."""
    attn_weights = np.array(attn_weights)
    values = np.array(values)
    mask = np.array(mask)

    if dropout_rate == 0:
        context = attn_weights @ values
        return context.tolist()

    dropped_attn = attn_weights * mask
    keep_prob = 1.0 - dropout_rate
    scale = 1.0 / keep_prob
    scaled_attn = scale * dropped_attn
    context = scaled_attn @ values
    return context.tolist()