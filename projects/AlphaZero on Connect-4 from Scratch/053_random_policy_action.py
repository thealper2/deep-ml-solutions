def random_policy_action(state, to_play, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    legal = valid_moves(state)
    return int(rng.choice(legal))
