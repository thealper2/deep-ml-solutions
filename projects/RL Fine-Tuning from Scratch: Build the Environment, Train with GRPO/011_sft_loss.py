import torch


def sft_loss(policy, batch):
    rows = [p + c for p, c in batch]
    seqs = torch.tensor(rows, dtype=torch.long)
    logp = completion_logprobs(policy, seqs)
    return -logp.mean()