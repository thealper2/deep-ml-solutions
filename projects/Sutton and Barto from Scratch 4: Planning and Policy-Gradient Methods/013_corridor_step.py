def corridor_step(state, action):
    if state == 1:
        action = 1 - action

    if action == 0:
        nxt = state - 1
    else:
        nxt = state + 1

    if nxt < 0:
        nxt = 0

    done = bool(nxt == 3)
    return (nxt, -1.0, done)