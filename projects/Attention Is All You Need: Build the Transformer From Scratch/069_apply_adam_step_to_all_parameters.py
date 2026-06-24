import torch

def apply_adam_step_to_all_parameters(parameter_list, optimizer_state, learning_rate, beta1=0.9, beta2=0.98, epsilon=1e-9):
    t = optimizer_state['t'] + 1
    optimizer_state['t'] = t

    m_list = optimizer_state['m']
    v_list = optimizer_state['v']

    for i, param in enumerate(parameter_list):
        if param.grad is None:
            continue

        grad = param.grad

        m_new = beta1 * m_list[i] + (1 - beta1) * grad
        m_list[i] = m_new

        v_new = beta2 * v_list[i] + (1 - beta2) * (grad ** 2)
        v_list[i] = v_new

        m_hat = m_new / (1 - beta1 ** t)
        v_hat = v_new / (1 - beta2 ** t)

        param.data -= learning_rate * m_hat / (torch.sqrt(v_hat) + epsilon)

    return optimizer_state
