def shift_targets_right_with_start_token(target_ids, start_token_id):
    batch_size, seq_len = target_ids.shape
    shifted = torch.full_like(target_ids, start_token_id)
    shifted[:, 1:] = target_ids[:, :-1]
    return shifted
