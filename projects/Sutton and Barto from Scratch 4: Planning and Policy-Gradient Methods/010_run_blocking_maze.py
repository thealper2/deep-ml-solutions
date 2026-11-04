import numpy as np


def run_blocking_maze(use_plus, max_steps, change_step, n_planning, kappa, alpha, gamma, epsilon, rng):
    walls_before = [(3, c) for c in range(8)]
    walls_after = [(3, c) for c in range(1, 9)]

    maze = build_maze(walls_before, start=(5, 3), goal=(0, 8))

    q_table = np.zeros((6, 9, 4))
    model = {}
    state = maze["start"]
    cumulative = np.zeros(max_steps, dtype=float)
    total = 0.0

    for t in range(1, max_steps + 1):
        if t == change_step + 1:
            maze = build_maze(walls_after, start=(5, 3), goal=(0, 8))

        action = epsilon_greedy_tabular(q_table, state, epsilon, rng)
        next_state, reward, done = maze_step(maze, state, action)

        q_table = q_learning_update(q_table, state, action, reward, next_state, alpha, gamma)

        if use_plus:
            model = update_model_plus(model, state, action, next_state, reward, t, 4)
            q_table = dyna_q_plus_planning(q_table, model, n_planning, t, kappa, alpha, gamma, rng)
        else:
            model[(state, action)] = (next_state, reward)
            q_table = planning_updates(q_table, model, n_planning, alpha, gamma, rng)

        total += reward
        cumulative[t - 1] = total

        if done:
            state = maze["start"]
        else:
            state = next_state

    return cumulative