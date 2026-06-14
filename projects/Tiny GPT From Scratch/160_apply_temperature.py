def apply_temperature(logits, temperature):
    """Scale logits by 1/temperature before softmax sampling."""
    scaled_logits = logits / temperature
    return scaled_logits
