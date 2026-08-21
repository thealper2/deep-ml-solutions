def frozen_stack_adaptive_demo(tokens, embedding_weight, blocks, mixer, source_layer, dest_layer, alpha, steps, lr, seed=0):
    """Frozen-stack demo: baseline vs fixed recirc vs trained adaptive NTP."""
    torch.manual_seed(seed)
    
    B, T = tokens.shape
    D = embedding_weight.shape[1]
    
    embeddings = embed_tokens(tokens, embedding_weight)
    
    baseline_residuals = run_layers(embeddings, blocks)
    baseline_h = baseline_residuals[-1]
    baseline_logits = tied_lm_head(baseline_h, embedding_weight)
    baseline_loss = ntp_loss(baseline_logits, tokens)
    
    fixed_residuals = sequential_prefill(embeddings, blocks, source_layer, dest_layer, alpha)
    fixed_h = fixed_residuals[-1]
    fixed_logits = tied_lm_head(fixed_h, embedding_weight)
    fixed_loss = ntp_loss(fixed_logits, tokens)
    
    for key, value in mixer.items():
        mixer[key] = value.detach().requires_grad_(True)
    
    optimizer = torch.optim.Adam(mixer.values(), lr=lr)
    base_residuals = run_layers(embeddings, blocks)
    
    for step in range(steps):
        s = base_residuals[source_layer].detach()
        d = base_residuals[dest_layer].detach()
        
        d_mixed = adaptive_recirculate(s, d, mixer)
        
        new_residuals = base_residuals.copy()
        new_residuals[dest_layer] = d_mixed
        
        x = new_residuals[dest_layer]
        for i in range(dest_layer, len(blocks)):
            x = pre_norm_block(x, blocks[i])
            new_residuals[i + 1] = x
        
        h = new_residuals[-1]
        logits = tied_lm_head(h, embedding_weight)
        loss = ntp_loss(logits, tokens)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        base_residuals = [r.detach() for r in new_residuals]
    
    residuals = run_layers(embeddings, blocks)
    
    s = residuals[source_layer]
    d = residuals[dest_layer]
    d_mixed = adaptive_recirculate(s, d, mixer)
    
    new_residuals = residuals.copy()
    new_residuals[dest_layer] = d_mixed
    
    x = new_residuals[dest_layer]
    for i in range(dest_layer, len(blocks)):
        x = pre_norm_block(x, blocks[i])
        new_residuals[i + 1] = x
    
    adaptive_h = new_residuals[-1]
    adaptive_logits = tied_lm_head(adaptive_h, embedding_weight)
    adaptive_loss = ntp_loss(adaptive_logits, tokens)
    
    return baseline_loss, fixed_loss, adaptive_loss