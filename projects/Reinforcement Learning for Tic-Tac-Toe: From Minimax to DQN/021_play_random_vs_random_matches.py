def play_random_vs_random_matches(n_games, rng):
    """Run n_games random-vs-random games and return the list of outcome strings."""
    results = []
    for _ in range(n_games):
        result = play_random_vs_random_game(rng)
        results.append(result)

    return results
