import numpy as np

def loop_fill_counts(n_matrix, data):
    """Increment n_matrix[curr, next] for every consecutive pair in data."""
    for i in range(len(data) - 1):
        n_matrix[data[i], data[i + 1]] += 1

    return n_matrix
