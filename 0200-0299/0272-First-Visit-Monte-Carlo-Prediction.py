import numpy as np

def first_visit_mc_prediction(
    episodes: list[list[tuple[int, float]]],
    n_states: int,
    gamma: float
) -> np.ndarray:
    """
    Estimate state values using First-Visit Monte Carlo prediction.
    
    Args:
        episodes: List of episodes. Each episode is a list of (state, reward) tuples.
                 The reward at index i is the reward received AFTER leaving state i.
        n_states: Number of states (states are integers 0 to n_states-1)
        gamma: Discount factor
        
    Returns:
        V: Estimated state values as numpy array of shape (n_states,)
    """
    total = np.zeros(n_states, dtype=float)
    count = np.zeros(n_states, dtype=int)

    for episode in episodes:
        G = 0.0
        returns = [0.0] * len(episode)
        for t in range(len(episode) - 1, -1, -1):
            _, reward = episode[t]
            G = reward + gamma * G
            returns[t] = G

        visited = set()
        for t, (state, _) in enumerate(episode):
            if state not in visited:
                visited.add(state)
                total[state] += returns[t]
                count[state] += 1

    V = np.zeros(n_states, dtype=float)
    for s in range(n_states):
        if count[s] > 0:
            V[s] = total[s] / count[s]
    
    return V