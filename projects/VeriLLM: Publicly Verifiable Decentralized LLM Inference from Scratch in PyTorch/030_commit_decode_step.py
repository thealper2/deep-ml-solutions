def commit_decode_step(step_state):
    step_index = hash_tensor(np.array([step_state['step_index']], dtype=np.int64))
    input_token = hash_tensor(np.array([step_state['input_token']], dtype=np.int64))
    next_token = hash_tensor(np.array([step_state['next_token']], dtype=np.int64))
    logits = hash_tensor(step_state['logits'])
    next_pos = hash_tensor(np.array([step_state['next_pos']], dtype=np.int64))

    kv_hashes = []
    for kv_cache in step_state['kv_caches']:
        k_hash = hash_tensor(kv_cache['k'])
        v_hash = hash_tensor(kv_cache['v'])
        kv_hashes.append(k_hash + v_hash)

    kv_combined = b''.join(kv_hashes)
    kv_combined_hash = hashlib.sha256(kv_combined).digest()

    combined = step_index + input_token + next_token + logits + kv_combined_hash + next_pos
    return hashlib.sha256(combined).digest()
