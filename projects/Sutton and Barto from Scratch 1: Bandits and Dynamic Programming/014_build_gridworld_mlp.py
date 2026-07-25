def build_gridworld_mdp():
    n_states = 16
    n_actions = 4
    n_rows = 4
    n_cols = 4
    terminal_states = {0, 15}

    P = {}
    for s in range(n_states):
        P[s] = {}
        for a in range(n_actions):
            if s in terminal_states:
                P[s][a] = [(1.0, s, 0.0)]
                continue
            
            row = s // n_cols
            col = s % n_cols

            if a == 0:
                next_row = max(0, row - 1)
                next_col = col
            elif a == 1:
                next_row = row
                next_col = min(n_cols - 1, col + 1)
            elif a == 2:
                next_row = min(n_rows - 1, row + 1)
                next_col = col
            else:
                next_row = row
                next_col = max(0, col - 1)

            next_state = next_row * n_cols + next_col
            reward = -1.0

            P[s][a] = [(1.0, next_state, reward)]

    return {
        'n_states': n_states,
        'n_actions': n_actions,
        'P': P,
    }
