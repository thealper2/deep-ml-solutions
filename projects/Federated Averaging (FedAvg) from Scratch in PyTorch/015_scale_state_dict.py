def scale_state_dict(state_dict, weight):
    result = {}
    for key in state_dict:
        result[key] = state_dict[key] * weight
        
    return result
