import numpy as np

def masked_ce_loss(logits: np.ndarray, targets: np.ndarray, mask: np.ndarray) -> float:
    """
    Compute mean cross-entropy loss over masked (response) positions only.

    Args:
        logits: (seq_len, vocab_size) array of unnormalized scores.
        targets: (seq_len,) array of integer target token ids.
        mask: (seq_len,) boolean array; True = include in loss.

    Returns:
        Mean cross-entropy over positions where mask is True (float).
    """
    if not np.any(mask):
        return 0.0
    
    max_logits = np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits - max_logits)
    sum_exp = np.sum(exp_logits, axis=1, keepdims=True)
    log_softmax = logits - max_logits - np.log(sum_exp)
    target_log_probs = log_softmax[np.arange(len(targets)), targets]
    masked_loss = -target_log_probs[mask]
    return float(np.mean(masked_loss))
