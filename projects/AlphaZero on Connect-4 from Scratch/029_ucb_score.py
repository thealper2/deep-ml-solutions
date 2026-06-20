import math

def ucb_score(parent, child, c_puct=1.5):
    q = node_q_value(child)
    exploration = c_puct * child['prior'] * np.sqrt(parent['visit_count']) / (1 + child['visit_count'])
    return q + exploration
