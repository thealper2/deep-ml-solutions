def pad_id_sequence(ids, max_len, pad_id):
    return ids + [pad_id] * (max_len - len(ids)) if max_len > len(ids) else ids[:max_len]
