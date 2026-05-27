import numpy as np

def generalized_policy_iteration(
    num_states: int,
    num_actions: int,
    transitions: np.ndarray,
    rewards: np.ndarray,
    discount: float,
    num_iterations: int,
    eval_sweeps: int,
    terminal_states: list
) -> dict:
    """
    Simulate Generalized Policy Iteration on a finite MDP.
    
    Args:
        num_states: Number of states in the MDP
        num_actions: Number of actions available
        transitions: Shape (S, A, S) transition probabilities
        rewards: Shape (S, A, S) reward function
        discount: Discount factor gamma in [0, 1]
        num_iterations: Number of GPI cycles (eval + improve)
        eval_sweeps: Number of synchronous policy evaluation sweeps per cycle
        terminal_states: List of terminal state indices
    
    Returns:
        Dictionary with:
            'values': List of floats (rounded to 4 decimal places)
            'policy': List of ints (action for each state)
    """
    V = np.zeros(num_states)
    policy = np.zeros(num_states, dtype=int)
    terminal_mask = np.zeros(num_states, dtype=bool)
    terminal_mask[terminal_states] = True

    for cycle in range(num_iterations):
        for sweep in range(eval_sweeps):
            V_new = V.copy()
            for s in range(num_states):
                if terminal_mask[s]:
                    V_new[s] = 0.0
                    continue

                a = policy[s]
                expected_value = 0.0
                for s_next in range(num_states):
                    p = transitions[s, a, s_next]
                    r = rewards[s, a, s_next]
                    expected_value += p * (r + discount * V[s_next])

                V_new[s] = expected_value

            V = V_new

        for s in range(num_states):
            if terminal_mask[s]:
                continue
            
            q_values = np.zeros(num_actions)
            for a in range(num_actions):
                q_val = 0.0
                for s_next in range(num_states):
                    p = transitions[s, a, s_next]
                    r = rewards[s, a, s_next]
                    q_val += p * (r + discount * V[s_next])

                q_values[a] = q_val

            best_action = np.argmax(q_values)
            policy[s] = best_action

    return {
        'values': np.round(V, 4).tolist(),
        'policy': policy.tolist(),
    }