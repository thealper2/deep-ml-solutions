def evaluate_planner(encoder_params, predictor_params, room_size, agent_size, n_episodes, max_steps, n_sequences, horizon, n_actions):
    successes = 0
    total_steps = 0
    total_final_distance = 0.0

    for episode in range(n_episodes):
        goal = torch.randint(0, room_size, (2,)).float()

        result = run_mpc_episode(
            encoder_params, predictor_params, goal, room_size, agent_size,
            max_steps, n_sequences, horizon, n_actions
        )

        if result['success']:
            successes += 1

        total_steps += result['steps']
        total_final_distance += result['final_distance']

    return {
        'success_rate': successes / n_episodes,
        'mean_steps': total_steps / n_episodes,
        'mean_final_distance': total_final_distance / n_episodes,
    }
