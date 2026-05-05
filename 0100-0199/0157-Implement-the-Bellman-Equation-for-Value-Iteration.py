import numpy as np

def bellman_update(V, transitions, gamma):
    """
    Perform one step of value iteration using the Bellman equation.
    Args:
      V: np.ndarray, state values, shape (n_states,)
      transitions: list of dicts. transitions[s][a] is a list of (prob, next_state, reward, done)
      gamma: float, discount factor
    Returns:
      np.ndarray, updated state values
    """
    n_states = len(V)
    new_V = np.zeros(n_states)

    for s in range(n_states):
      best_value = -np.inf
      for a, outcomes in transitions[s].items():
        q_value = 0.0
        for prob, next_state, reward, done in outcomes:
          if done:
            future_value = 0.0
          else:
            future_value = V[next_state]

          q_value += prob * (reward + gamma * future_value)

        best_value = max(best_value, q_value)

      new_V[s] = best_value

    return new_V