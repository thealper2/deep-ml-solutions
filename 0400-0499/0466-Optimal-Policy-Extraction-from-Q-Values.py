import numpy as np

def extract_optimal_policy(Q: np.ndarray) -> dict:
    """
    Extract the optimal policy, state-value function, and advantage
    function from a Q-value table.
    
    Args:
        Q: Q-value table of shape (num_states, num_actions)
    
    Returns:
        Dictionary with keys:
        - 'optimal_actions': list of int (optimal action per state)
        - 'state_values': list of float (V*(s) per state)
        - 'advantages': nested list of float (A(s,a) for all pairs)
    """
    num_states, num_actions = Q.shape

    optimal_actions = []
    state_values = []
    advantages = []

    for state in range(num_states):
        q_values = Q[state]

        best_action = int(np.argmax(q_values))
        best_value = float(q_values[best_action])

        optimal_actions.append(best_action)
        state_values.append(round(best_value, 4))

        state_advantages = [round(q - best_value, 4) for q in q_values]
        advantages.append(state_advantages)

    return {
        "optimal_actions": optimal_actions,
        "state_values": state_values,
        "advantages": advantages,
    }
