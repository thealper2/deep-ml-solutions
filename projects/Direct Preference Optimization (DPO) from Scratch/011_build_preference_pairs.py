def build_preference_pairs(prompts, chosen_ids, rejected_ids, chosen_mask, rejected_mask):
    pairs = []
    N = len(prompts)
    for i in range(N):
        pairs.append({
            'prompt': prompts[i],
            'chosen_ids': chosen_ids[i].tolist(),
            'rejected_ids': rejected_ids[i].tolist(),
            'chosen_mask': chosen_mask[i].tolist(),
            'rejected_mask': rejected_mask[i].tolist(),
        })

    return pairs
