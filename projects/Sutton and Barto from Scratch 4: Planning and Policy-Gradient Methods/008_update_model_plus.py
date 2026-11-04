def update_model_plus(model, state, action, next_state, reward, time, n_actions=4):
    m = dict(model)
    for a in range(n_actions):
        key = (state, a)
        if key not in m:
            m[key] = (state, 0.0, 0)

    m[(state, action)] = (next_state, reward, time)
    return m