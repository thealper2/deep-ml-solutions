import numpy as np


def fly(kind, n_flights, rng, start=START):
    states = []
    actions = []
    rewards = []
    next_states = []
    dones = []
    flights = []

    for fidx in range(n_flights):
        state = tuple(int(x) for x in start)
        while True:
            probs = autopilot_probs(state, kind)
            cdf = np.cumsum(probs)
            action = min(int(np.searchsorted(cdf, rng.random(), side="right")), 2)
            next_state, reward, done = lander_step(state, action, rng)

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
            flights.append(fidx)

            state = next_state
            if done:
                break

    states = np.array(states, dtype=int)
    actions = np.array(actions, dtype=int)
    rewards = np.array(rewards, dtype=float)
    next_states = np.array(next_states, dtype=int)
    dones = np.array(dones, dtype=bool)
    flights = np.array(flights, dtype=int)

    returns = np.zeros(len(rewards), dtype=float)
    running = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        running = rewards[t] + running
        returns[t] = running
        if t == 0 or dones[t - 1]:
            running = 0.0

    return {
        "states": states,
        "actions": actions,
        "rewards": rewards,
        "next_states": next_states,
        "dones": dones,
        "flight": flights,
        "returns": returns,
    }