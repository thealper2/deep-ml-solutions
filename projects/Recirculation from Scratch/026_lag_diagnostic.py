def lag_diagnostic(embeddings, tokens, blocks, embedding_weight, t, k, source_layer, dest_layer, alpha):
    """Change in next-token log-likelihood at lag k after recirculating position t."""
    B, T, D = embeddings.shape
    baseline_residuals = run_layers(embeddings, blocks)
    baseline_h = baseline_residuals[-1]
    baseline_logits = tied_lm_head(baseline_h, embedding_weight)
    baseline_log_probs = F.log_softmax(baseline_logits, dim=-1)
    pred_idx = t + k
    target_idx = pred_idx + 1
    
    if pred_idx >= T - 1 or target_idx >= T:
        return torch.tensor(0.0, device=embeddings.device)
    
    baseline_ll = baseline_log_probs[:, pred_idx, :].gather(1, tokens[:, target_idx:target_idx+1])
    residuals = run_layers(embeddings, blocks)
    recirculated_residuals = recirculate_one_position(
        residuals, t, source_layer, dest_layer, alpha, blocks
    )
    recirculated_h = recirculated_residuals[-1]
    recirculated_logits = tied_lm_head(recirculated_h, embedding_weight)
    recirculated_log_probs = F.log_softmax(recirculated_logits, dim=-1)
    recirculated_ll = recirculated_log_probs[:, pred_idx, :].gather(1, tokens[:, target_idx:target_idx+1])
    delta = (recirculated_ll - baseline_ll).mean()    
    return delta