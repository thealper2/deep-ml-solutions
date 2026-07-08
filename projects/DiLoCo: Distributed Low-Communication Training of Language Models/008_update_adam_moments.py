def update_adam_moments(state, grads, beta1, beta2):
    state['t'] += 1

    for key in grads:
        state['m'][key] = beta1 * state['m'][key] + (1 - beta1) * grads[key]
        state['v'][key] = beta2 * state['v'][key] + (1 - beta2) * (grads[key] ** 2)

    return state
