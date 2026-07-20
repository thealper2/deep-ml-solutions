def rollout_latent_dynamics(initial_embedding: torch.Tensor, actions: torch.Tensor, predictor_params: dict) -> torch.Tensor:
    if initial_embedding.dim() == 1:
        T = actions.shape[0]
        current = initial_embedding.unsqueeze(0)
        trajectory = [current]
        
        for t in range(T):
            action = actions[t].unsqueeze(0)
            current = predict_next_embedding(current, action, predictor_params)
            trajectory.append(current)
        
        return torch.cat(trajectory, dim=0)
    else:
        if actions.dim() == 1:
            T = actions.shape[0]
            B = initial_embedding.shape[0]
            actions_expanded = actions.unsqueeze(0).expand(B, -1)
        else:
            B, T = actions.shape
            actions_expanded = actions
        
        current = initial_embedding
        trajectory = [current]
        
        for t in range(T):
            action = actions_expanded[:, t]
            current = predict_next_embedding(current, action, predictor_params)
            trajectory.append(current)
        
        return torch.stack(trajectory, dim=0)
