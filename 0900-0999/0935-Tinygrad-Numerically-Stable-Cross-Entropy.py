from tinygrad import Tensor

def cross_entropy(logits, targets):
    logits_max = logits.max(axis=-1, keepdim=True)
    logits_shifted = logits - logits_max
    log_sum_exp = logits_shifted.exp().sum(axis=-1, keepdim=True).log()
    log_probs = logits_shifted - log_sum_exp
    batch_indices = Tensor.arange(targets.shape[0])
    target_log_probs = log_probs[batch_indices, targets]
    nll = -target_log_probs
    return nll.mean()