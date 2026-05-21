import numpy as np

def tanh_soft_cap(logits, softcap):
    """Apply tanh soft-capping to logits.

    Args:
        logits: numpy array of any shape.
        softcap: positive float, or None/<=0 to disable.

    Returns:
        numpy array of the same shape with soft-capped values,
        rounded to 6 decimal places.
    """
    if softcap is None or softcap <= 0:
        return logits

    out = softcap * np.tanh(logits / softcap)
    out = np.round(out, 6)
    return out
