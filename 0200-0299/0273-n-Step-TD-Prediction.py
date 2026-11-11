import numpy as np

def n_step_td_prediction(
    episodes: list[list[tuple[int, float]]],
    n_states: int,
    n: int,
    gamma: float,
    alpha: float
) -> np.ndarray:
    """
    Perform n-step TD prediction to estimate state values.
    
    Args:
        episodes: List of episodes. Each episode is a list of (state, reward) tuples.
                 The reward at index i is the reward received AFTER leaving state i.
                 Episodes end with a terminal transition (last state's reward is the final reward).
        n_states: Number of states (states are integers 0 to n_states-1)
        n: Number of steps for n-step TD
        gamma: Discount factor
        alpha: Learning rate
        
    Returns:
        V: Estimated state values as numpy array of shape (n_states,)
    """
    V = np.zeros(n_states, dtype=float)
    
    for episode in episodes:
        T = len(episode)
        for t in range(T):
            G = 0.0
            for k in range(n):
                if t + k >= T:
                    break
                G += (gamma ** k) * episode[t + k][1]

            if t + n < T:
                G += (gamma ** n) * V[episode[t + n][0]]
            s = episode[t][0]
            V[s] += alpha * (G - V[s])
    
    return V