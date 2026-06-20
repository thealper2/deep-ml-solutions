import torch

def masked_policy_logits(logits, mask):
    """Set logits at illegal columns to -inf.

    logits: torch.Tensor of shape (..., 7)
    mask:   bool array/tensor of shape (7,), True = legal
    returns: torch.Tensor of same shape as logits
    """
    if isinstance(mask, np.ndarray):
        mask = torch.tensor(mask, dtype=torch.bool)

    if isinstance(logits, np.ndarray):
        logits = torch.tensor(logits)

    result = logits.clone()

    if result.dim() == 2 and mask.dim() == 1:
        mask = mask.unsqueeze(0).expand(result.shape[0], -1)
    
    result[~mask] = float('-inf')
    return result
