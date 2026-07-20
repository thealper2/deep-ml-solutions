def run_mpc_episode(encoder_params, predictor_params, goal_pos, room_size, agent_size, max_steps, n_sequences, horizon, n_actions):
    if isinstance(goal_pos, tuple):
        goal_pos = torch.tensor(goal_pos, dtype=torch.float32)
    elif isinstance(goal_pos, list):
        goal_pos = torch.tensor(goal_pos, dtype=torch.float32)

    state, obs = env_reset(room_size, seed=None)
    trajectory = [(state[0].item(), state[1].item())]

    goal_embedding = encode_goal(goal_pos, encoder_params, room_size)

    success = False
    steps = 0

    for step in range(max_steps):
        current_obs = render_observation(state, room_size)
        current_embedding = encode_batch(current_obs.unsqueeze(0),encoder_params).squeeze(0)
        action = mpc_step(current_embedding, goal_embedding, predictor_params, n_sequences, horizon, n_actions)
        state, obs = env_step(state, action, room_size)
        trajectory.append((state[0].item(), state[1].item()))
        steps += 1

        if torch.allclose(state, goal_pos, atol=0.5):
            success = True
            break

    final_distance = torch.sqrt(torch.sum((state - goal_pos) ** 2)).item()

    return {
        'success': success,
        'steps': steps,
        'trajectory': trajectory,
        'final_distance': final_distance,
    }
