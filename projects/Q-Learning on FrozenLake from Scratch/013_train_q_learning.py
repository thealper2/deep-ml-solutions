import numpy as np

def train_q_learning(env, num_episodes, alpha=0.1, gamma=0.99, epsilon_start=1.0, epsilon_min=0.01, epsilon_decay=0.999, seed=0, max_steps=200):
    rng = np.random.default_rng(seed)

    env.action_space.seed(seed)
    env.reset(seed=seed)

    n_states = env.observation_space.n
    n_actions = env.action_space.n

    q_table = np.zeros((n_states, n_actions), dtype=float)
    episode_returns = []

    epsilon = epsilon_start

    for episode in range(num_episodes):
        state, _ = env.reset(seed=seed + episode)
        state = int(state)

        total_reward = 0.0

        for _ in range(max_steps):

            if rng.random() < epsilon:
                action = int(env.action_space.sample())
            else:
                action = int(np.argmax(q_table[state]))

            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = int(next_state)

            done = terminated or truncated

            best_next = 0.0 if done else np.max(q_table[next_state])
            td_target = reward + gamma * best_next

            q_table[state, action] += alpha * (td_target - q_table[state, action])

            state = next_state
            total_reward += float(reward)

            if done:
                break

        episode_returns.append(total_reward)

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    return q_table, episode_returns
