def maze_step(maze, state, action):
    deltas = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
    dr, dc = deltas[action]
    r, c = state[0] + dr, state[1] + dc

    if 0 <= r < maze["rows"] and 0 <= c < maze["cols"] and (r, c) not in maze["walls"]:
        next_state = (r, c)
    else:
        next_state = state

    reward = 1.0 if next_state == maze["goal"] else 0.0
    done = bool(next_state == maze["goal"])
    return (next_state, reward, done)