import numpy as np

def q_learning(num_states, num_actions, P, R, terminal_states, alpha, gamma, epsilon, num_episodes):
    Q = np.zeros((num_states, num_actions))
    terminal_set = set(terminal_states)

    for episode in range(num_episodes):
        start_state = np.random.choice([s for s in range(num_states) if s not in terminal_set])
        state = start_state

        while state not in terminal_set:
            if np.random.random() < epsilon:
                action = np.random.randint(num_actions)
            else:
                action = np.argmax(Q[state])

            next_state = np.random.choice(num_states, p=P[state, action])
            reward = R[state, action]
            td_target = reward + gamma * np.max(Q[next_state])
            td_error = td_target - Q[state, action]
            Q[state, action] += alpha * td_error
            state = next_state

    return Q