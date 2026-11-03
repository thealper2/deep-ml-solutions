def draw_card(rng):
    rank = int(rng.integers(1, 14))
    return min(rank, 10)