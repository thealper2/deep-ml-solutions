import numpy as np
from sklearn.linear_model import LogisticRegression

def propagate_and_train(X_train, y_train, kmeans, rep_idx, X_test, y_test, percentile=20):
    distances = kmeans.transform(X_train)
    labels = kmeans.labels_
    d = distances[np.arange(len(X_train)), labels]

    selected_idx = []
    propagated_labels = []

    for j in range(len(rep_idx)):
        members = np.where(labels == j)[0]
        cutoff = np.percentile(d[members], percentile)
        chosen = members[d[members] <= cutoff]
        selected_idx.extend(chosen)
        propagated_labels.extend([y_train[rep_idx[j]]] * len(chosen))

    selected_idx = np.array(selected_idx)
    propagated_labels = np.array(propagated_labels)

    label_accuracy = float(np.mean(propagated_labels == y_train[selected_idx]))

    model = LogisticRegression(max_iter=10000)
    model.fit(X_train[selected_idx], propagated_labels)
    test_accuracy = float(model.score(X_test, y_test))

    return {
        "n_propagated": int(len(selected_idx)),
        "label_accuracy": label_accuracy,
        "test_accuracy": test_accuracy,
    }