def apply_log_softmax_over_vocab(logits):
    return torch.log_softmax(logits, dim=-1)
