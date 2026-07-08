def adam_param_step(params, m_hat, v_hat, lr, eps):
    new_params = {}
    for key in params:
        new_params[key] = params[key] - lr * m_hat[key] / (np.sqrt(v_hat[key]) + eps)

    return new_params
