def jepa_world_model_experiment(room_size, agent_size, embed_dim, n_train_transitions, n_epochs, batch_size, n_probe_samples, n_eval_episodes, max_steps, n_sequences, horizon):
    torch.manual_seed(0)
    dataset = build_transition_dataset(n_train_transitions, room_size, seed=0)
    encoder_params = init_encoder_params(obs_channels=1, room_size=room_size, latent_dim=embed_dim, seed=0)
    target_params = init_target_encoder(encoder_params)
    predictor_params = init_predictor_params(latent_dim=embed_dim, action_dim=4, hidden_dim=64, seed=0)
    encoder_params, target_params, predictor_params, history = train_jepa(
        dataset, encoder_params, target_params, predictor_params,
        num_steps=n_epochs, batch_size=batch_size, lr=1e-3, tau=0.99, seed=0
    )
    train_losses = [h['loss'] for h in history]
    collapse_metrics = [h['collapse'] for h in history]
    probe_result = probe_state_recovery(dataset, encoder_params, num_probe_steps=100)
    mse = probe_result['mse']
    states = dataset['states']
    var = torch.mean((states - torch.mean(states, dim=0, keepdim=True)) ** 2) + 1e-8
    probe_r2 = 1 - mse / var.item()
    eval_result = evaluate_planner(
        encoder_params, predictor_params, room_size, agent_size,
        n_eval_episodes, max_steps, n_sequences, horizon, 4
    )

    return {
        'train_losses': train_losses,
        'collapse_metrics': collapse_metrics,
        'probe_r2': probe_r2,
        'success_rate': eval_result['success_rate'],
        'mean_steps': eval_result['mean_steps'],
    }
