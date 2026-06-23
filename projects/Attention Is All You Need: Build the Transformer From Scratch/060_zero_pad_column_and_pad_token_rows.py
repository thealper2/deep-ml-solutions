import torch

def zero_pad_column_and_pad_token_rows(smoothed_distribution, gold_token_ids, pad_id):
    out = smoothed_distribution.clone()
    out[:, :, pad_id] = 0.0
    pad_mask = (gold_token_ids == pad_id)
    out[pad_mask] = 0.0
    return out
