def extract_greedy_policy(q_table):
    return np.argmax(q_table, axis=1).astype(np.int64)
