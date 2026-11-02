import torch


def kl_k3(logp, ref_logp):
    d = ref_logp - logp
    return torch.exp(d) - d - 1.0