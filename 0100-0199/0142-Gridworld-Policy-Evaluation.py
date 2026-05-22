import numpy as np

def gridworld_policy_evaluation(policy: dict, gamma: float, threshold: float) -> list[list[float]]:
    """
    Evaluate state-value function for a policy on a 5x5 gridworld.
    
    Args:
        policy: dict mapping (row, col) to action probability dicts
        gamma: discount factor
        threshold: convergence threshold
    Returns:
        5x5 list of floats
    """
    V = [[0.0 for _ in range(5)] for _ in range(5)]
    terminal_states = {(0, 0), (0, 4), (4, 0), (4, 4)}
    moves = {
        'up': (-1, 0),
        'down': (1, 0),
        'left': (0, -1),
        'right': (0, 1),
    }

    while True:
        delta = 0
        V_new = [row[:] for row in V]

        for i in range(5):
            for j in range(5):
                if (i, j) in terminal_states:
                    V_new[i][j] = 0.0
                    continue
                
                expected_value = 0.0
                for action, prob in policy[(i, j)].items():
                    di, dj = moves[action]
                    ni, nj = i + di, j + dj
                    if not (0 <= ni < 5 and 0 <= nj < 5):
                        ni, nj = i, j

                    expected_value += prob * (-1 + gamma * V[ni][nj])

                V_new[i][j] = expected_value
                delta = max(delta, abs(V_new[i][j] - V[i][j]))

        V = V_new
        if delta < threshold:
            break

    return V