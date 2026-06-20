import torch

def greedy_action_from_policy(logits, mask):
    """Return the argmax legal column index from masked policy logits."""
    masked = masked_policy_logits(logits, mask)
    return int(torch.argmax(masked).item())
