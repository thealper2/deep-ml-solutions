def add_state_dicts(state_a, state_b):
    result = {}
    for key in state_a:
        result[key] = state_a[key] + state_b[key]
        
    return result
