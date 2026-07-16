def freeze_reference_logprobs(ref_params, pairs):
    out = []
    for pair in pairs:
        chosen_ids = np.array(pair['chosen_ids']).reshape(1, -1)
        chosen_mask = np.array(pair['chosen_mask']).reshape(1, -1)
        rejected_ids = np.array(pair['rejected_ids']).reshape(1, -1)
        rejected_mask = np.array(pair['rejected_mask']).reshape(1, -1)
        chosen_logprob = policy_sequence_logprob(ref_params, chosen_ids, chosen_mask)[0]
        rejected_logprob = policy_sequence_logprob(ref_params, rejected_ids, rejected_mask)[0]
        out.append({
            'chosen': chosen_logprob,
            'rejected': rejected_logprob,
        })

    return out
