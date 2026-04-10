import numpy as np

def compare_policy_evaluation(V: np.ndarray, P: np.ndarray, R: np.ndarray, gamma: float, n_sweeps: int, method: str) -> np.ndarray:
    """
    Perform policy evaluation sweeps using the specified update method.
    
    Args:
        V: np.ndarray of shape (n_states,), initial value function
        P: np.ndarray of shape (n_states, n_states), transition probability matrix under the policy
        R: np.ndarray of shape (n_states,), expected immediate reward for each state
        gamma: float, discount factor
        n_sweeps: int, number of sweeps to perform
        method: str, either 'synchronous' or 'in_place'
    
    Returns:
        np.ndarray: updated value function after n_sweeps
    """
    n_states = len(V)
    V_current = V.copy()

    if method == 'synchronous':
        for _ in range(n_sweeps):
            V_new = np.zeros(n_states)
            for s in range(n_states):
                V_new[s] = R[s] + gamma * np.sum(P[s, :] * V_current)

            V_current = V_new

    elif method == 'in_place':
        for _ in range(n_sweeps):
            for s in range(n_states):
                V_current[s] = R[s] + gamma * np.sum(P[s, :] * V_current)

    return V_current
