def random_walk_episode(rng):
    episode = []
    state = 3

    while state not in (0, 6):
        u = rng.random()
        next_state = state + 1 if u < 0.5 else state - 1

        reward = 1.0 if next_state == 6 else 0.0
        episode.append((int(state), float(reward), int(next_state)))

        state = next_state

    return episode