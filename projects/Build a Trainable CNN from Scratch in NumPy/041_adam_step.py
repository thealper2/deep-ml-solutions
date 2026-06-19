import numpy as np

def adam_step(param, grad, m, v, t, lr, beta_one, beta_two, eps):
    new_m = adam_update_m(m, grad, beta_one)
    new_v = adam_update_v(v, grad, beta_two)

    m_hat = adam_bias_correct(new_m, beta_one, t)
    v_hat = adam_bias_correct(new_v, beta_two, t)

    new_param = adam_param_step(param, m_hat, v_hat, lr, eps)

    return new_param, new_m, new_v
