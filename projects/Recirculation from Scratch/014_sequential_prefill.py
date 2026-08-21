def sequential_prefill(embeddings, blocks, source_layer, dest_layer, alpha, ramp_steps=10):
    """Token-by-token recirculation prefill with ramped alpha."""
    B, T, D = embeddings.shape
    x = torch.zeros_like(embeddings)
    residuals = run_layers(x, blocks)

    for t in range(T):
        prefix = embeddings[:, :t+1, :]
        prefix_residuals = run_layers(prefix, blocks)
        current_residuals = []

        for layer_idx in range(len(prefix_residuals)):
            layer_full = residuals[layer_idx]
            layer_prefix = prefix_residuals[layer_idx]
            combined = layer_full.clone()
            combined[:, t:t+1, :] = layer_prefix[:, t:t+1, :]
            current_residuals.append(combined)
        
        alpha_t = ramped_alpha(t, alpha, ramp_steps)
        current_residuals = recirculate_one_position(
            current_residuals, t, source_layer, dest_layer, alpha_t, blocks
        )
        residuals = current_residuals
    
    return residuals