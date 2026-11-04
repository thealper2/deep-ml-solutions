import numpy as np
import heapq


def prioritized_sweeping_episode(maze, q_table, model, n_planning, theta, alpha, gamma, epsilon, rng):
    m = dict(model)
    queue = []
    state = maze["start"]
    n_steps = 0
    n_updates = 0
    q = q_table

    while True:
        action = epsilon_greedy_tabular(q, state, epsilon, rng)
        next_state, reward, done = maze_step(maze, state, action)
        m[(state, action)] = (next_state, reward)
        push_priority(queue, q, state, action, reward, next_state, gamma, theta)

        for _ in range(n_planning):
            if not queue:
                break

            _, s, a = heapq.heappop(queue)
            s_next, r = m[(s, a)]
            q = q_learning_update(q, s, a, r, s_next, alpha, gamma)
            n_updates += 1
            for (ps, pa), (pn, pr) in m.items():
                if pn == s:
                    push_priority(queue, q, ps, pa, pr, s, gamma, theta)

        n_steps += 1
        state = next_state
        if done:
            break

    return q, m, n_steps, n_updates