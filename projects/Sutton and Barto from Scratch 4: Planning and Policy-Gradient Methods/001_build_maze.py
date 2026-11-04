def build_maze(walls=None, start=(2, 0), goal=(0, 8), rows=6, cols=9):
    if walls is None:
        walls = [(1, 2), (2, 2), (3, 2), (4, 5), (0, 7), (1, 7), (2, 7)]
        
    return {
        "rows": int(rows),
        "cols": int(cols),
        "start": tuple(start),
        "goal": tuple(goal),
        "walls": {(int(r), int(c)) for r, c in walls},
        "n_actions": 4,
    }