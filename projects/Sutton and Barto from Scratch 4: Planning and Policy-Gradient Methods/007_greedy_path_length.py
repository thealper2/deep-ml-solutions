import numpy as np


def greedy_path_length(maze, q_table, max_steps=100):
    state = maze["start"]
    for step in range(1, max_steps + 1):
        action = int(np.argmax(q_table[state]))
        next_state, _, done = maze_step(maze, state, action)
        if done:
            return step

        state = next_state

    return max_steps