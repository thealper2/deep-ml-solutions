def multi_step_prediction_error(dataset: dict, encoder_params: dict, target_params: dict, predictor_params: dict, horizon: int = 5, num_samples: int = 32) -> float:
    obs = dataset['observations']
    actions = dataset['actions']
    next_obs = dataset['next_observations']
    N = obs.shape[0]
    max_start = N - horizon
    if max_start <= 0:
        return 0.0

    num_samples = min(num_samples, max_start)
    total_loss = 0.0
    total_steps = 0

    for i in range(num_samples):
        start_obs = obs[i].unsqueeze(0)
        current_embedding = encode_batch(start_obs, encoder_params)
        future_actions = actions[i:i+horizon]
        predicted_traj = rollout_latent_dynamics(current_embedding.squeeze(0), future_actions, predictor_params)
        predicted_future = predicted_traj[1:]
        future_obs = next_obs[i:i+horizon]
        true_future = encode_batch(future_obs, target_params)
        loss = torch.mean((predicted_future - true_future) ** 2)
        total_loss += loss.item()
        total_steps += 1

    return total_loss / total_steps if total_steps > 0 else 0.0
