def length_normalized_logprob(seq_logprob, mask):
    sequence_lengths = np.sum(mask, axis=1)
    return seq_logprob / sequence_lengths
