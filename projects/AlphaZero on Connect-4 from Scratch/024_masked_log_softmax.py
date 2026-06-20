import torch
import torch.nn.functional as F

def masked_log_softmax(logits, mask):
    """Log-softmax of logits with illegal columns (mask=False) forced to -inf."""
    masked_logits = masked_policy_logits(logits, mask)
    return F.log_softmax(masked_logits, dim=-1)
