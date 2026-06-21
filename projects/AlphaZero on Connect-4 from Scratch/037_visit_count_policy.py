def visit_count_policy(root, temperature=1.0):
    pi = np.zeros(7, dtype=np.float32)
    
    if not root['children']:
        pi[:] = 1.0 / 7.0
        return np.round(pi, 4).tolist()
    
    visits = np.zeros(7, dtype=np.float32)
    for action, child in root['children'].items():
        visits[action] = child['visit_count']
    
    visited_actions = [action for action, child in root['children'].items() if child['visit_count'] > 0]
    
    if not visited_actions:
        pi[:] = 1.0 / 7.0
        return np.round(pi, 4).tolist()
    
    if temperature == 0.0:
        best_action = np.argmax(visits)
        pi[best_action] = 1.0
    else:
        visit_counts = np.array([visits[a] for a in visited_actions])
        
        if temperature != 1.0:
            weighted = visit_counts ** (1.0 / temperature)
        else:
            weighted = visit_counts
        
        probs = weighted / np.sum(weighted)
        
        for action, prob in zip(visited_actions, probs):
            pi[action] = prob
    
    return np.round(pi, 4).tolist()
