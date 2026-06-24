import torch

def average_loss_over_non_pad_tokens(total_loss, gold_token_ids, pad_id):
    non_pad_mask = (gold_token_ids != pad_id)
    num_non_pad = non_pad_mask.sum()
    if num_non_pad == 0:
        return total_loss

    return total_loss / num_non_pad
