def replace_pair(sequences, pair, new_id):
    new_sequences = []
    for seq in sequences:
        new_sequence = []
        i = 0
        while i < len(seq):
            p = seq[i:i+len(pair)]
            if p == list(pair):
                i += len(pair)
                new_sequence.append(new_id)
            else:
                new_sequence.append(seq[i])
                i += 1

        new_sequences.append(new_sequence)

    return new_sequences