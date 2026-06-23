def apply_dropout_with_keep_mask(x, keep_mask, keep_prob):
    return x * keep_mask / keep_prob
