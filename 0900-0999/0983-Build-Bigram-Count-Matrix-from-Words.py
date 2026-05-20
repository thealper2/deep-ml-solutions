import numpy as np

def bigram_counts(words):
    unique_chars = set(''.join(words + ['.']))
    unique_chars = sorted(unique_chars)
    stoi = {v: k for k, v in enumerate(unique_chars)}
    V = len(unique_chars)
    N = np.zeros((V, V), dtype=int)
    for word in words:
        new_word = '.' + word + '.'
        n = len(new_word)
        for i in range(n - 1):
            p = new_word[i:i+2]
            idx1 = stoi[p[0]]
            idx2 = stoi[p[1]]
            N[idx1][idx2] += 1

    return N.tolist()
