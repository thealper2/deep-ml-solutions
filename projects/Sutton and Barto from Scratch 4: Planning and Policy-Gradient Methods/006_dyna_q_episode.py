import numpy as np


def dyna_q_episode(maze, q_table, model, n_planning, alpha, gamma, epsilon, rng):
    m = dict(model)
    q = q_table
    state = maze["start"]
    steps = 0

    while True:
        action = epsilon_greedy_tabular(q, state, epsilon, rng)
        next_state, reward, done = maze_step(maze, state, action)
        q = q_learning_update(q, state, action, reward, next_state, alpha, gamma)
        m[(state, action)] = (next_state, reward)
        q = planning_updates(q, m, n_planning, alpha, gamma, rng)

        steps += 1
        state = next_state
        if done:
            break

    return q, m, steps