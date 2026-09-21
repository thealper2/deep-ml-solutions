import numpy as np

def gambler_value_iteration(ph, theta=1e-9):
    """
    Computes the optimal value function and policy for the Gambler's Problem.
    Args:
      ph: probability of heads
      theta: convergence threshold
    Returns:
      V: list of values for all states 0..100
      policy: list of optimal stakes for all states 0..100
    """
    goal = 100
    V = np.zeros(goal + 1)
    V[goal] = 1.0

    while True:
        delta = 0
        for s in range(1, goal):
            old_v = V[s]
            actions = np.arange(1, min(s, goal - s) + 1)
            action_values = []
            for a in actions:
                val = ph * V[s + a] + (1 - ph) * V[s - a]
                action_values.append(val)

            V[s] = np.max(action_values)
            delta = max(delta, abs(old_v - V[s]))

        if delta < theta:
            break

        policy = np.zeros(goal + 1, dtype=int)
        for s in range(1, goal):
            actions = np.arange(1, min(s, goal - s) + 1)
            action_values = [ph * V[s + a] + (1 - ph) * V[s - a] for a in actions]
            policy[s] = actions[np.argmax(np.round(action_values, 4))]

    return V, policy