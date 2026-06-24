import torch

def pick_next_token_by_argmax(final_step_logits):
    """Greedy: return argmax token id per batch row.

    final_step_logits: FloatTensor of shape (batch, vocab_size)
    returns: LongTensor of shape (batch,)
    """
    return torch.argmax(final_step_logits, dim=-1)
