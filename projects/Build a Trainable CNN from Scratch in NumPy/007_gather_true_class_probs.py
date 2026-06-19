def gather_true_class_probs(probs, labels):
    return np.array([prob[label] for prob, label in zip(probs, labels)])
