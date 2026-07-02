def leaf_prediction(labels):
    counts = np.bincount(labels)
    return int(np.argmax(counts))
