def aggregate_weighted_average(client_states, client_sample_counts):
    total_samples = sum(client_sample_counts)
    weighted_states = []

    for state, count in zip(client_states, client_sample_counts):
        weight = count / total_samples
        weighted_states.append(scale_state_dict(state, weight))

    result = weighted_states[0]
    for i in range(1, len(weighted_states)):
        result = add_state_dicts(result, weighted_states[i])

    return result
