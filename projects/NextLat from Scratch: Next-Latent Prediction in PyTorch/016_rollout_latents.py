def rollout_latents(h, x, params: dict, dyn: dict, d_steps: int) -> list:
    B, T, d = h.shape
    wte = params['wte']
    
    h_hat = h[:, :T - d_steps]
    predictions = []
    
    for i in range(1, d_steps + 1):
        token_indices = x[:, i:T - d_steps + i]
        emb = wte[token_indices]
        
        h_hat = latent_transition(h_hat, emb, dyn)
        predictions.append(h_hat)
    
    return predictions