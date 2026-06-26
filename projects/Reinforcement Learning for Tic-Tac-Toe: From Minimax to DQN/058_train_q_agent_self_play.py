def train_q_agent_self_play(num_episodes, alpha, gamma, initial_epsilon, min_epsilon, decay_rate, rng):
    q_table = initialize_q_table()
    episode_outcomes = []

    for episode in range(num_episodes):
        epsilon = epsilon_decay_schedule(initial_epsilon, episode, min_epsilon, decay_rate)

        result = self_play_episode(q_table, alpha, gamma, epsilon, rng)
        final_status = result['final_status']
        transitions = result['transitions']

        for transition in transitions:
            state_key = transition['state_key']
            action = transition['action']
            reward = transition['reward']
            next_board = transition['next_board']
            done = transition['done']
            player = transition['player']

            flipped_state_key = canonical_board_key(flip_board_perspective(next_board, player))
            flipped_reward = perspective_reward_sign(reward, player, 1)
            episode_apply_q_update(q_table, state_key, action, flipped_reward, next_board, done, alpha, gamma)

        episode_outcomes.append(final_status)

    return {
        'q_table': q_table,
        'episode_outcomes': episode_outcomes,
    }
