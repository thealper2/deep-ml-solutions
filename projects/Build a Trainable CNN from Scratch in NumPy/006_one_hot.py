def one_hot(labels, num_classes):
    one_hot = np.eye(num_classes)[labels]
    return one_hot
