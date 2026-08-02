import numpy as np

def solve_sanity_env(transitions: dict, gamma: float, tol: float = 1e-6, max_iter: int = 10000) -> dict:
    """
    Solve a deterministic MDP using value iteration.
    
    Args:
        transitions: dict mapping state -> {action: (next_state, reward, done)}
        gamma: float, discount factor
        tol: float, convergence tolerance
        max_iter: int, maximum iterations
    
    Returns:
        dict with 'V' (state values rounded to 4 decimals) and 'policy' (optimal actions)
    """
    states = list(transitions.keys())
    
    V = {s: 0.0 for s in states}
    
    for _ in range(max_iter):
        delta = 0.0
        V_new = {}
        
        for s in states:
            q_values = []
            for action, (next_state, reward, done) in transitions[s].items():
                if done:
                    q = reward
                else:
                    q = reward + gamma * V[next_state]
                q_values.append((q, action))
            
            best_q = max(q_values, key=lambda x: x[0])[0]
            V_new[s] = best_q
            
            delta = max(delta, abs(V_new[s] - V[s]))
        
        V = V_new
        
        if delta < tol:
            break
    
    policy = {}
    for s in states:
        q_values = []
        for action, (next_state, reward, done) in transitions[s].items():
            if done:
                q = reward
            else:
                q = reward + gamma * V[next_state]
            q_values.append((q, action))
        
        best_action = min(q_values, key=lambda x: (-x[0], x[1]))[1]
        policy[s] = best_action
    
    V = {s: round(v, 4) for s, v in V.items()}
    return {'V': V, 'policy': policy}
