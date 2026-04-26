import numpy as np

def td_prediction(episodes: list, n_states: int, gamma: float, alpha: float, mode: str) -> list:
    """
    Perform TD(0) prediction in online or offline mode.
    
    Args:
        episodes: List of episodes, each a list of (state, reward, next_state, done) tuples
        n_states: Number of states
        gamma: Discount factor
        alpha: Learning rate
        mode: 'online' or 'offline'
    
    Returns:
        List of estimated state values rounded to 4 decimal places
    """
    V = np.zeros(n_states, dtype=np.float64)

    if mode == "online":
        for episode in episodes:
            for state, reward, next_state, done in episode:
                target = reward + gamma * V[next_state] if not done else reward
                td_error = target - V[state]
                V[state] += alpha * td_error

    elif mode == "offline":
        for episode in episodes:
            V_start = V.copy()
            updates = [0.0] * n_states
            for state, reward, next_state, done in episode:
                target = reward + gamma * V_start[next_state] if not done else reward
                td_error = target - V_start[state]
                updates[state] += alpha * td_error

            for s in range(n_states):
                V[s] += updates[s]

    else:
        raise ValueError("Invalid mode")

    return [round(v, 4) for v in V]