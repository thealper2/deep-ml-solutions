def sample_from_neural_bigram(w, start_id, num_tokens, itos):
    """Generate a string by repeatedly sampling from softmax of W[id]."""
    rng = np.random.default_rng(0)
    ids = [start_id]
    current_id = start_id

    for _ in range(num_tokens):
        logits = forward_logits_lookup(w, np.array([current_id]))
        probs = logits_to_probs_rowwise(logits)[0]
        next_id = rng.choice(len(probs), p=probs)
        ids.append(next_id)
        current_id = next_id

    return decode_ids(ids, itos)
