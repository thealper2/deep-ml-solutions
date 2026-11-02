import torch


def completion_logprobs(policy, seqs):
    logits = policy(seqs[:, :-1])
    logp = torch.log_softmax(logits, dim=-1)
    target = seqs[:, PROMPT_LEN:]
    sliced = logp[:, PROMPT_LEN - 1:, :]
    gathered = sliced.gather(2, target.unsqueeze(-1)).squeeze(-1)
    return gathered