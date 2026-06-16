def init_adam_state(params):
    m = {}
    v = {}
    
    for key, arr in params.items():
        m[key] = np.zeros_like(arr)
        v[key] = np.zeros_like(arr)
    
    return {'m': m, 'v': v, 't': 0}
