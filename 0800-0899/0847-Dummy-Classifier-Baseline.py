import numpy as np
from collections import Counter

def dummy_classifier(y_train, n_test, strategy, constant=None):
    """
    Produce baseline predictions of length n_test using the given strategy.
    Returns a Python list of predicted labels.
    """
    if strategy == 'most_frequent':
        counts = Counter(y_train)
        most_freq = max(sorted(counts.keys()), key=lambda x: counts[x])
        return [most_freq] * n_test

    elif strategy == 'constant':
        if constant is None:
            raise ValueError('')

        return [constant] * n_test

    elif strategy == 'uniform':
        unique = sorted(set(y_train))
        k = len(unique)
        return [unique[i % k] for i in range(n_test)]

    elif strategy == 'stratified':
        counts = Counter(y_train)
        n_train = len(y_train)
        unique = sorted(counts.keys())

        base_counts = {}
        fractions = {}
        for c in unique:
            exact = counts[c] * n_test / n_train
            base_counts[c] = int(exact)
            fractions[c] = exact - base_counts[c]

        remaining = n_test - sum(base_counts.values())

        if remaining > 0:
            sorted_classes = sorted(unique, key=lambda x: (-fractions[x], x))
            for i in range(remaining):
                base_counts[sorted_classes[i]] += 1

        results = []
        for c in unique:
            results.extend([c] * base_counts[c])

        return results

    else:
        raise ValueError('Unknown strategy')