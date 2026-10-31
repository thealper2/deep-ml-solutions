import numpy as np


def top_state(n):
    return np.ones((n, n, n), dtype=bool)


def encode_givens(grid):
    grid = np.asarray(grid)
    n = grid.shape[0]
    x = np.ones((n, n, n), dtype=bool)
    for r in range(n):
        for c in range(n):
            d = grid[r, c]
            if d != 0:
                x[r, c, :] = False
                x[r, c, d - 1] = True

    return x


def meet(a, b):
    return np.logical_and(a, b)


def join(a, b):
    return np.logical_or(a, b)


def leq(a, b):
    return not np.any(a & ~b)

def alive_count(x):
    return int(np.sum(x))


def is_bottom(x):
    return bool(np.any(np.sum(x, axis=2) == 0))


def is_solved(x):
    return bool(np.all(np.sum(x, axis=2) == 1))


def to_grid(x):
    x = np.asarray(x)
    n = x.shape[0]
    counts = np.sum(x, axis=2)
    grid = np.zeros((n, n), dtype=int)
    for r in range(n):
        for c in range(n):
            if counts[r, c] == 1:
                grid[r, c] = int(np.argmax(x[r, c])) + 1

    return grid