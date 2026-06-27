def append_kv_cache(kv_cache, new_k, new_v):
    if kv_cache['k'] is None:
        kv_cache['k'] = new_k
        kv_cache['v'] = new_v
    else:
        kv_cache['k'] = np.concatenate([kv_cache['k'], new_k], axis=0)
        kv_cache['v'] = np.concatenate([kv_cache['v'], new_v], axis=0)
        
    return kv_cache
