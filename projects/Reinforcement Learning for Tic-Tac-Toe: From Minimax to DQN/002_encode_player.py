def encode_player(player):
    """Return the integer encoding for 'X', 'O', or 'empty'."""
    d = {'X': 1, 'O': -1, 'empty': 0}
    return d[player]
