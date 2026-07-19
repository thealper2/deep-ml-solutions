def predictor_forward(embeddings: torch.Tensor, actions: torch.Tensor, predictor_params: dict) -> torch.Tensor:
    action_emb = embed_action(actions, predictor_params)
    x = torch.cat([embeddings, action_emb], dim=-1)
    x = x @ predictor_params['fc1_w'].T + predictor_params['fc1_b']
    x = torch.relu(x)
    x = x @ predictor_params['fc2_w'].T + predictor_params['fc2_b']    
    return x
