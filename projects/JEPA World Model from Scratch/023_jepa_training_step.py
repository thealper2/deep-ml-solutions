def jepa_training_step(batch: dict, encoder_params: dict, target_params: dict, predictor_params: dict, lr: float = 1e-3, tau: float = 0.99) -> tuple[dict, dict, dict, float, float]:
    obs = batch['observations']
    actions = batch['actions']
    next_obs = batch['next_observations']
    for key in encoder_params:
        encoder_params[key].requires_grad_(True)

    for key in predictor_params:
        predictor_params[key].requires_grad_(True)
    
    online_embeddings = encode_batch(obs, encoder_params)
    target_embeddings = encode_batch(next_obs, target_params).detach()
    predicted = predict_next_embedding(online_embeddings, actions, predictor_params)
    loss = jepa_loss(predicted, target_embeddings, online_embeddings)
    loss.backward()
    with torch.no_grad():
        for key in encoder_params:
            if encoder_params[key].grad is not None:
                encoder_params[key].data = encoder_params[key].data - lr * encoder_params[key].grad.data
                encoder_params[key].grad = None
        
        for key in predictor_params:
            if predictor_params[key].grad is not None:
                predictor_params[key].data = predictor_params[key].data - lr * predictor_params[key].grad.data
                predictor_params[key].grad = None
    
    target_params = ema_update(target_params, encoder_params, tau)
    col = collapse_metric(online_embeddings).item()
    return encoder_params, target_params, predictor_params, loss.item(), col
