def compute_batch_training_loss(src_batch, tgt_batch, model_params, config):
    pad_id = config['pad_id']
    start_id = config['start_id']
    vocab_size = config['vocab_size']
    smoothing = config.get('smoothing', 0.1)
    num_heads = config['num_heads']
    
    if 'token_embedding' not in model_params:
        model_params['token_embedding'] = model_params['tgt_embedding']
    
    decoder_input = shift_targets_right_with_start_token(tgt_batch, start_id)
    
    log_probs = run_transformer_forward(src_batch, decoder_input, model_params, num_heads, pad_id)
    
    batch_size, seq_len, _ = log_probs.shape
    if smoothing == 0.0:
        smoothed_dist = torch.zeros_like(log_probs)
        smoothed_dist.scatter_(-1, tgt_batch.unsqueeze(-1), 1.0)
    else:
        uniform_val = smoothing / (vocab_size - 2)
        smoothed_dist = torch.full_like(log_probs, uniform_val)
        confidence = 1.0 - smoothing
        smoothed_dist = set_confidence_on_gold_tokens(smoothed_dist, tgt_batch, confidence)
    
    smoothed_dist = zero_pad_column_and_pad_token_rows(smoothed_dist, tgt_batch, pad_id)
    
    total_loss = compute_label_smoothed_kl_loss(log_probs, smoothed_dist)
    
    loss = average_loss_over_non_pad_tokens(total_loss, tgt_batch, pad_id)
    
    return loss
