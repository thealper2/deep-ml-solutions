import torch

def set_confidence_on_gold_tokens(smoothed_distribution, gold_token_ids, confidence):
    """Place confidence mass at gold-token positions of a smoothed target distribution."""
    out = smoothed_distribution.clone()
    batch_size, seq_len, vocab_size = out.shape
    batch_indices = torch.arange(batch_size).unsqueeze(1).expand(batch_size, seq_len)
    seq_indices = torch.arange(seq_len).unsqueeze(0).expand(batch_size, seq_len)
    out[batch_indices, seq_indices, gold_token_ids] = confidence
    return out
