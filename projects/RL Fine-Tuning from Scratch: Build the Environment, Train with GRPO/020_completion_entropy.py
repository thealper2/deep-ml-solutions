import torch


@torch.no_grad()
def completion_entropy(policy, seqs, mask):
    logits = policy(seqs[:, :-1])
    sliced = logits[:, PROMPT_LEN - 1:, :]
    logp = torch.log_softmax(sliced, dim=-1)
    p = logp.exp()
    H = -(p * logp).sum(dim=-1)
    return float((mask * H).sum() / mask.sum())