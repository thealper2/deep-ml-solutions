def compute_attention_scores(queries, keys):
    return queries @ keys.T
