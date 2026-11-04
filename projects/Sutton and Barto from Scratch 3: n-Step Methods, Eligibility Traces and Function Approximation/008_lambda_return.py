import numpy as np

def lambda_return(rewards, states, t, lam, values, gamma=1.0):
    T = len(rewards)
    n_max = T - t
    G = 0.0
    for n in range(1, n_max):
        G += (1.0 - lam) * (lam ** (n - 1)) * n_step_return(rewards, states, t, n, values, gamma)
        
    G += (lam ** (n_max - 1)) * n_step_return(rewards, states, t, n_max, values, gamma)
    return float(G)