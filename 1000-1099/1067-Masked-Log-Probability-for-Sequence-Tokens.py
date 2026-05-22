import numpy as np
import numpy.ma as ma

def masked_avg_logprob(logits, labels, selection_mask):
    batch, num_tokens, vocab_size = logits.shape
    max_logits = np.max(logits, axis=-1, keepdims=True)
    log_probs = logits - max_logits - np.log(np.sum(np.exp(logits - max_logits), axis=-1, keepdims=True))
    indices = np.expand_dims(labels, axis=-1)
    selected_logprobs = np.take_along_axis(log_probs, indices, axis=-1).squeeze(axis=-1)
    masked_logprobs = selected_logprobs * selection_mask
    total_selected = np.sum(selection_mask)
    if total_selected == 0:
        return 0.0

    return float(np.sum(masked_logprobs) / total_selected)