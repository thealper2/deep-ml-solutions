def kda_recurrence(q, k, v, alpha, beta, S0=None):
    """Run KDA token by token: update state, then read O[t] = S_t^T q[t].

    Returns (O, S_final) with O of shape (T, dv). S0 defaults to zeros; never
    mutate the caller's S0.
    """
    T, dk = q.shape
    dv = v.shape[1]

    if S0 is None:
        S = np.zeros((dk, dv))
    else:
        S = S0.copy()

    O = np.zeros((T, dv))

    for t in range(T):
        S = kda_state_update(S, k[t], v[t], alpha[t], beta[t])
        O[t] = S.T @ q[t]

    return O, S
