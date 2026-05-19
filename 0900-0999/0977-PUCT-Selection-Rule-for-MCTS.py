import numpy as np

def puct_select(Q, N_children, P, N_parent, c_puct):
    """Return the index of the child maximizing the PUCT score."""
    Q = np.array(Q)
    N_children = np.array(N_children)
    P = np.array(P)
    exploration = c_puct * P * (np.sqrt(N_parent) / (1 + N_children))
    scores = Q + exploration
    return int(np.argmax(scores))