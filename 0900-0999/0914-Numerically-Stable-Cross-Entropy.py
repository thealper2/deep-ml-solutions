import torch

def cross_entropy(logits, targets):
    logits_max = logits.max(dim=-1, keepdim=True)[0]
    logits_shifted = logits - logits_max
    log_sum_exp = torch.log(torch.exp(logits_shifted).sum(dim=-1, keepdim=True))
    log_probs = logits_shifted - log_sum_exp
    nll = -log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    return nll.mean()