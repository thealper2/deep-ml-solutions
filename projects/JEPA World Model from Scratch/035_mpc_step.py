def mpc_step(start_embedding, goal_embedding, predictor_params, n_sequences, horizon, n_actions):
    action_sequences = sample_action_sequences(n_sequences, horizon, n_actions)
    costs = score_action_sequences(start_embedding, action_sequences, goal_embedding, predictor_params)
    best_plan = select_best_plan(action_sequences, costs)
    return int(best_plan[0].item())
