def crop_context_to_block_size(context_ids, block_size):
    T = context_ids.shape[1]
    return context_ids if T <= block_size else context_ids[:, -block_size:]
