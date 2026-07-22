import numpy as np

def td0_prediction(episodes: list, n_states: int, gamma: float = 0.99, alpha: float = 0.01) -> list:
    """
    Perform TD(0) prediction to estimate the state-value function.
    
    Args:
        episodes: List of episodes, each episode is a list of
                  (state, reward, next_state, done) tuples
        n_states: Total number of states
        gamma: Discount factor
        alpha: Learning rate / step size
    
    Returns:
        List of estimated state values
    """
    V = [0.0] * n_states

    for episode in episodes:
        for state, reward, next_state, done in episode:
            if done:
                td_target = reward
            else:
                td_target = reward + gamma * V[next_state]

            td_error = td_target - V[state]
            V[state] += alpha * td_error

    return V
