def sample_average_action_values(k: int, actions: list, rewards: list) -> tuple:
    """
    Estimate action values using sample averaging.
    
    Args:
        k: Number of possible actions (labeled 0 to k-1)
        actions: List of actions taken at each time step
        rewards: List of rewards received at each time step
        
    Returns:
        Tuple of (Q, N) where Q is estimated values and N is selection counts
    """
    Q = []
    N = []
    for k_action in range(k):
        arr = []
        for a, b in zip(actions, rewards):
            if a == k_action:
                arr.append(b)

        n = len(arr)
        Q_val = sum(arr) / n if n else 0.0
        Q.append(round(Q_val, 4))
        N.append(n)

    return Q, N
