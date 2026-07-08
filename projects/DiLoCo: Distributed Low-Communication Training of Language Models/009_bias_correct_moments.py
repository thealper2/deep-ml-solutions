def bias_correct_moments(state, beta1, beta2):
    t = state['t']
    m_hat = {}
    v_hat = {}
    for key in state['m']:
        m_hat[key] = state['m'][key] / (1 - beta1 ** t)
        v_hat[key] = state['v'][key] / (1 - beta2 ** t)

    return m_hat, v_hat
