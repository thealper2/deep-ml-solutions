import numpy as np

def column_full(board, column):
    """Return True if `column` has no empty rows left."""
    return True if column_top_row(board, column) == -1 else False
